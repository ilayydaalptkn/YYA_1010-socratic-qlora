from pathlib import Path

SRC = Path(r"C:\Users\ilayd\OneDrive\Masaüstü\socratic-debugging-benchmark-main\socratic_debugging_benchmark\v1_bea\final_dataset")
OUT = Path(r"C:\Users\ilayd\OneDrive\Masaüstü\socratic_qlora\data\train_seed.jsonl")

print("SRC exists:", SRC.exists())
print("txt count:", len(list(SRC.glob("*.txt"))))
print("OUT will be:", OUT)
sample = next(SRC.glob("*.txt"))
text = sample.read_text(encoding="utf-8", errors="replace")
print("sample file:", sample.name)
print("--- first 500 chars ---")
print(text[:500])
start = text.find("<dialogue>")
end = text.find("</dialogue>")
print("dialogue found:", start != -1 and end != -1)
if start != -1 and end != -1:
    dialogue = text[start + len("<dialogue>"):end].strip()
    print("--- dialogue first 400 chars ---")
    print(dialogue[:400])
    import re

# <alt>...</alt> bloklarını sil (tek satır alt'lar için)
clean = re.sub(r"<alt>.*", "", dialogue)
# <code>...</code> şimdilik düz metin kalsın; etiketleri silelim
clean = re.sub(r"</?code>", "", clean)

messages = []
for line in clean.splitlines():
    line = line.strip()
    if not line:
        continue
    if line.startswith("User:"):
        messages.append({"role": "user", "content": line[len("User:"):].strip()})
    elif line.startswith("Assistant:"):
        messages.append({"role": "assistant", "content": line[len("Assistant:"):].strip()})

print("message count:", len(messages))
print("first 2 messages:")
for m in messages[:2]:
    print(m["role"], ":", m["content"][:80])
    import json

OUT.parent.mkdir(parents=True, exist_ok=True)
n_ok = 0

with OUT.open("w", encoding="utf-8") as f:
    for path in sorted(SRC.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        s = raw.find("<dialogue>")
        e = raw.find("</dialogue>")
        if s == -1 or e == -1:
            continue
        dlg = raw[s + len("<dialogue>"):e]
        dlg = re.sub(r"<alt>.*", "", dlg)
        dlg = re.sub(r"</?code>", "", dlg)

        msgs = []
        for line in dlg.splitlines():
            line = line.strip()
            if line.startswith("User:"):
                msgs.append({"role": "user", "content": line[len("User:"):].strip()})
            elif line.startswith("Assistant:"):
                msgs.append({"role": "assistant", "content": line[len("Assistant:"):].strip()})

        if len(msgs) < 2:
            continue

        f.write(json.dumps({"messages": msgs, "source": path.name}, ensure_ascii=False) + "\n")
        n_ok += 1

print("written examples:", n_ok)
print("saved to:", OUT)