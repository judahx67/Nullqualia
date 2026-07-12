"""Feed lora.py's own output back into itself as a two-sided conversation.

Not two model instances -- one model, one adapter (that's the whole point of a
style model), just alternating whose turn it is to seed the next generate() call.

CLI: python self_talk.py [n_turns] [seed message]
"""
import sys

from lora import generate

N_TURNS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED = " ".join(sys.argv[2:]) or "hey"


def main():
    msg = SEED
    print(f"A: {msg}")
    for i in range(N_TURNS):
        msg = generate(msg)
        print(f"{'B' if i % 2 == 0 else 'A'}: {msg}")


if __name__ == "__main__":
    main()
