"""Phase 2b: LoRA fine-tune of Llama-3.2-1B on data/corpus.json for style-only continuation.

No (context -> reply) pairs exist in this export, so this trains on raw message
continuation (packed causal-LM), not instruction-following. Harness contract
(all model tiers): generate(prompt=None) -> str.

CLI: python lora.py train   |   python lora.py chat   |   python lora.py test   |   python lora.py [n_samples]

Needs HF_TOKEN (Llama-3.2 is gated on Hugging Face). Copy .env.example to .env
and fill it in yourself -- lora.py loads it automatically.
"""
import json
import os
import sys
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.2-1B"
ROOT = Path(__file__).parent
ADAPTER_DIR = ROOT / "data" / "lora-adapter"


def _load_dotenv(path=ROOT / ".env"):
    """Minimal .env loader (no python-dotenv dep) so HF_TOKEN etc. don't need to be
    exported by hand. Populate .env yourself, per .env.example; it's gitignored."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

CORPUS = ROOT / "data" / "corpus.json"
BLOCK_SIZE = 512  # packed sequence length; raise if VRAM allows, lower if OOM
BATCH_SIZE = 2  # calibration knob for the 4060's actual free VRAM
GRAD_ACCUM = 8
EPOCHS = 10  # 3 epochs at r=16 barely nudged the base model's pretraining prior; needs more signal


def pack_messages(messages, encode, eos_id, block_size=BLOCK_SIZE):
    """Concatenate encode(message)+eos for every message, then chunk into fixed-size
    blocks. Drops the trailing partial block (negligible loss at this corpus size)."""
    ids = []
    for m in messages:
        ids.extend(encode(m))
        ids.append(eos_id)
    return [ids[i:i + block_size] for i in range(0, len(ids) - block_size + 1, block_size)]


def _dataset(chunks):
    import torch

    class PackedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(chunks)

        def __getitem__(self, i):
            ids = torch.tensor(chunks[i], dtype=torch.long)
            return {"input_ids": ids, "labels": ids.clone()}

    return PackedDataset()


def train():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

    token = os.environ.get("HF_TOKEN")
    messages = json.loads(CORPUS.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    chunks = pack_messages(messages, lambda m: tok(m, add_special_tokens=False)["input_ids"], tok.eos_token_id)
    print(f"{len(messages)} messages -> {len(chunks)} packed blocks of {BLOCK_SIZE} tokens")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=token, torch_dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM", r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(ROOT / "data" / "lora-checkpoints"),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
    )
    Trainer(model=model, args=args, train_dataset=_dataset(chunks), data_collator=default_data_collator).train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tok.save_pretrained(str(ADAPTER_DIR))
    print(f"adapter saved to {ADAPTER_DIR}")


_state = {}


def _load():
    if "model" not in _state:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.environ.get("HF_TOKEN")
        tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=token, torch_dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        model.eval()
        _state["model"], _state["tok"] = model, tok
    return _state["model"], _state["tok"]


def generate(prompt=None):
    import torch

    model, tok = _load()
    text = prompt or ""
    # training packs <message><EOS><message><EOS>...; seed with EOS (the learned
    # message-boundary token), not BOS, or the base model's pretraining prior wins
    ids = [[tok.eos_token_id] + tok(text, add_special_tokens=False)["input_ids"]]
    input_ids = torch.tensor(ids).to(model.device)
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, min_new_tokens=3, do_sample=True, temperature=0.9,
                              top_p=0.95, pad_token_id=tok.eos_token_id)
    full = tok.decode(out[0][1:], skip_special_tokens=True)
    return full[len(text):].strip() or full.strip()


def chat():
    print("nullqualia (lora) ready. ctrl-c to quit.")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        print(generate(prompt or None))


def test():
    encode = lambda s: [ord(c) for c in s]  # fake tokenizer: 1 "token" per char
    chunks = pack_messages(["ab", "cd", "efg"], encode, eos_id=0, block_size=3)
    # "ab"+eos, "cd"+eos, "efg"+eos -> [97,98,0, 99,100,0, 101,102,103, 0] len 10 -> 3 full blocks of 3
    assert chunks == [[97, 98, 0], [99, 100, 0], [101, 102, 103]], chunks
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == "chat":
        chat()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        for _ in range(n):
            print(generate())
