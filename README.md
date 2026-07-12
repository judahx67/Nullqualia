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

4. Generate:

   ```
   python markov.py         # 10 samples
   python markov.py 25      # 25 samples
   python markov.py test    # self-check
   ```

`Input/` and `data/` are gitignored (personal data) — only the empty folders are tracked.

## Model harness

Every model tier is one module exposing `generate(prompt=None) -> str`; the Discord bot imports whichever. `markov.py` rebuilds its chain from `data/corpus.json` at startup (~2s for ~117k messages). A `prompt` seeds the chain from a bigram containing one of the prompt's keywords — fake responsiveness, since a Markov chain can't actually reply.

## Notes

- Only my own messages are used as model outputs. The personal data package contains only those anyway; Phase 2's (incoming message → reply) pairs will need a different export.
- `data/corpus.json` is the single versioned dataset all tiers train on, so comparisons stay fair.
