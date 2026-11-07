from pathlib import Path
from email import policy
from email.parser import BytesParser
import pandas as pd
BASE = Path(__file__).resolve().parent
RAW = BASE / "dataset" / "raw"
OUT = BASE / "dataset" / "email_dataset.csv"
FOLDERS = {"easy_ham": "ham", "easy_ham_2": "ham", "spam": "spam"}

def read_email(path):
    try:
        with path.open("rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        subject = str(msg.get("subject", "") or "")
        parts = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        value = part.get_content()
                        if isinstance(value, str): parts.append(value)
                    except Exception: pass
        else:
            try:
                value = msg.get_content()
                if isinstance(value, str): parts.append(value)
            except Exception: pass
        return (subject + "\\n" + "\\n".join(parts)).strip()
    except Exception as exc:
        print(f"Skipping {path.name}: {exc}")
        return ""

rows = []
for folder, label in FOLDERS.items():
    path = RAW / folder
    if not path.exists():
        print(f"WARNING: folder not found: {path}")
        continue
    print(f"Reading {folder}...")
    for item in path.iterdir():
        if item.is_file():
            text = read_email(item)
            if len(text) >= 10: rows.append({"label": label, "message": text})
df = pd.DataFrame(rows, columns=["label", "message"])
if df.empty:
    raise SystemExit("No emails found. Copy extracted files into dataset/raw/easy_ham, easy_ham_2 and spam.")
df.drop_duplicates(subset=["message"], inplace=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False, encoding="utf-8")
print("\\nDataset prepared!")
print("Total emails:", len(df))
print(df["label"].value_counts().to_string())
print("Saved:", OUT)
