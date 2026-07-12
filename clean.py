"""Phase 0: merge Input/c*/messages.json into data/corpus.json, a JSON array of cleaned messages."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
MENTION = re.compile(r"<(@[!&]?|#)\d+>")
EMOJI = re.compile(r"<a?:\w+:\d+>")
URL = re.compile(r"https?://\S+")
# ponytail: char/keyword heuristic; accentless Viet slang still slips through — swap in lingua/langdetect if it matters
VIET = re.compile(
    r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]"
    r"|\b(khong|duoc|nguoi|minh|vcl|vkl|dcm|cmm)\b",
    re.IGNORECASE,
)

# ponytail: whole-message drop on slur hit; per-word masking if the corpus shrinks too much
SLURS = [w.strip().lower() for w in (ROOT / "slurs.txt").read_text(encoding="utf-8").splitlines()
         if w.strip() and not w.startswith("#")]
SLUR_RE = re.compile(r"\b(" + "|".join(map(re.escape, SLURS)) + r")\b", re.IGNORECASE) if SLURS else None


def clean(text: str) -> str:
    text = MENTION.sub("", text)
    text = EMOJI.sub("", text)
    text = URL.sub("", text)
    return " ".join(text.split())  # collapse whitespace/newlines; keep typos and casing


def main():
    messages = []  # (ID, text)
    for f in sorted(ROOT.glob("Input/c*/messages.json")):
        for m in json.loads(f.read_text(encoding="utf-8")):
            messages.append((m["ID"], m["Contents"]))

    messages.sort()  # chronological (snowflake IDs are time-ordered)
    kept, slurred, empty, viet = [], 0, 0, 0
    for _, raw in messages:
        text = clean(raw)
        if not text:
            empty += 1
        elif VIET.search(text):
            viet += 1
        elif SLUR_RE and SLUR_RE.search(text):
            slurred += 1
        else:
            kept.append(text)

    out = ROOT / "data" / "corpus.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(kept, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(messages)} messages -> {len(kept)} kept ({empty} empty after clean, {viet} Vietnamese, {slurred} dropped by slur filter)")
    print(f"wrote {out}")


def test():
    assert clean("hi <@123> and <@!456> in <#789>") == "hi and in"
    assert clean("sad <:sadge:1302927725086117888> ok <a:spin:123>") == "sad ok"
    assert clean("see https://example.com/x?y=1 now") == "see now"
    assert clean("  \n\t ") == ""
    assert clean("giữ nguyên typo VÀ CASING :v") == "giữ nguyên typo VÀ CASING :v"
    assert VIET.search("tính tự hàn mà :v")
    assert VIET.search("khong biet")  # accentless keyword
    assert not VIET.search("KEEP TYPOS and casing lol")
    print("ok")


if __name__ == "__main__":
    test() if "test" in sys.argv else main()
