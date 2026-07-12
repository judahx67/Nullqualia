"""Order-2 word-level Markov chain over data/corpus.json.

Harness contract (all model tiers): generate(prompt=None) -> str.
CLI: python markov.py [n_samples]   |   python markov.py chat   |   python markov.py test
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

START = "<S>"
END = "<E>"
MAX_WORDS = 60  # length cap so a loopy chain can't run forever


def build(messages):
    chain = defaultdict(Counter)
    for msg in messages:
        words = msg.split()
        if not words:
            continue
        tokens = [START, START] + words + [END]
        for i in range(len(tokens) - 2):
            chain[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1
    return chain


def _sample(counter):
    return random.choices(list(counter), weights=counter.values())[0]


def generate(chain, prompt=None):
    """Generate one message. If prompt given, seed from a bigram containing
    one of its words (the parlor-trick responsiveness from the plan)."""
    state = (START, START)
    if prompt:
        # ponytail: linear scan over keys per reply; index words->bigrams if the bot feels slow
        keywords = {w.lower() for w in prompt.split() if len(w) > 3}
        candidates = [k for k in chain if {k[0].lower(), k[1].lower()} & keywords]
        if candidates:
            state = random.choice(candidates)
    out = [w for w in state if w != START]
    while len(out) < MAX_WORDS:
        nxt = _sample(chain[state])
        if nxt == END:
            break
        out.append(nxt)
        state = (state[1], nxt)
    return " ".join(out)


def load_chain(corpus_path=Path(__file__).parent / "data" / "corpus.json"):
    return build(json.loads(Path(corpus_path).read_text(encoding="utf-8")))


def test():
    chain = build(["hello world foo", "hello world bar", "solo"])
    assert chain[(START, START)]["hello"] == 2
    assert set(chain[("hello", "world")]) == {"foo", "bar"}
    assert chain[("world", "foo")][END] == 1
    random.seed(0)
    for _ in range(20):
        msg = generate(chain)
        assert msg in {"hello world foo", "hello world bar", "solo"}, msg
    # prompt seeding lands on a bigram containing the keyword
    assert "world" in generate(chain, prompt="world domination")
    print("ok")


def chat():
    chain = load_chain()
    print("nullqualia ready. ctrl-c/ctrl-d to quit.")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        print(generate(chain, prompt=prompt or None))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    elif len(sys.argv) > 1 and sys.argv[1] == "chat":
        chat()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        chain = load_chain()
        for _ in range(n):
            print(generate(chain))
