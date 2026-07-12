# Nullqualia

A replica of me/you based off discord messages (realisitcally any form of messages). 
Include Markov chain and eventually maybe a LoRA fine-tune. 
Discord bot idea is in the brewing, I'm not sure about turning it into a bot. 

Python 3.12, stdlib-only so far.

## Getting started

1. Request your **Discord personal data package** (User Settings → Data & Privacy). Unzip its `messages/` channel folders (`c<channel_id>/` each containing `channel.json` + `messages.json`) into `Input/`.
2. (Optional) put slurs to filter in `slurs.txt`, one per line — matched whole-word, case-insensitive, drops the whole message.
3. Build the corpus:

   ```
   python clean.py          # writes data/corpus.json
   python clean.py test     # self-check
   ```

   Cleaning strips mentions, custom emoji codes, and URLs, drops empty/Vietnamese/slur messages, and deliberately **keeps typos and casing** — style fidelity is the point.

4. Generate (Markov, Phase 1):

   ```
   python markov.py         # 10 samples
   python markov.py 25      # 25 samples
   python markov.py chat    # interactive: your message seeds the reply
   python markov.py test    # self-check
   ```

5. LoRA fine-tune (Phase 2b), in `.venv` (`python -m venv .venv`, then `pip install torch --index-url https://download.pytorch.org/whl/cu124`, `pip install "transformers==4.46.3" "peft==0.13.2" accelerate numpy` — pinned because current `peft`/`transformers` need a newer torch than what's CUDA-compatible here):

   ```
   cp .env.example .env      # fill in HF_TOKEN yourself; lora.py loads it, I don't read it
   python lora.py train      # ~80min on a laptop 4060, writes data/lora-adapter/
   python lora.py            # 10 samples
   python lora.py chat       # interactive
   python lora.py test       # self-check (offline, no model download)
   ```

   `meta-llama/Llama-3.2-1B` is gated — accept the license on its HF page first.

`Input/` and `data/` are gitignored (personal data) — only the empty folders are tracked.

## Model harness

Every model tier is one module exposing `generate(prompt=None) -> str`; the Discord bot imports whichever.

- `markov.py`: rebuilds its chain from `data/corpus.json` at startup (~2s for ~117k messages). A `prompt` seeds the chain from a bigram containing one of the prompt's keywords — fake responsiveness, since a Markov chain can't actually reply.
- `lora.py`: no (context → reply) pairs exist in this export, so this is a style-only causal-LM fine-tune (LoRA, r=32) on packed raw messages, not instruction-following. A `prompt` is used as a literal continuation prefix; same "can't actually reply" caveat as Markov. Generation seeds from `EOS` rather than `BOS` since that's the message-boundary token the packed training data actually taught the model to condition on.

## Notes

- Only my own messages are used as model outputs. The personal data package contains only those anyway; Phase 2's (incoming message → reply) pairs will need a different export.
- `data/corpus.json` is the single versioned dataset all tiers train on, so comparisons stay fair.
