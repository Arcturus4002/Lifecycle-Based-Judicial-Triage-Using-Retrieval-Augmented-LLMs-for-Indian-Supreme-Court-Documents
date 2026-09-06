"""
build_jsonl_local.py
--------------------
Builds dataset_from_csv.jsonl from your local PDFs + classification_report.csv.
Upload the output JSONL to Drive — Colab will use it directly, skipping all PDF work.

Usage:
    pip install pdfplumber pandas scikit-learn tqdm
    python build_jsonl_local.py

Output:
    dataset_from_csv.jsonl  (same folder as this script)
"""

import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pdfplumber
from tqdm import tqdm

# ── CONFIG — edit these paths ─────────────────────────────────────────────────
PDF_ROOT  = r"C:\Users\udaya\Downloads\Project_2_Final\supreme_court_judgments"
CSV_PATH  = r"C:\Users\udaya\Downloads\Project_2_Final\classification_report.csv"
OUT_FILE  = r"C:\Users\udaya\Downloads\Project_2_Final\legal_training_data\dataset_from_csv.jsonl"

MAX_WORDS   = 1500   # truncate each judgment to this many words
MAX_WORKERS = 32     # threads — local SSD can handle high concurrency
MIN_CHARS   = 100    # skip PDFs with less text than this

# ── LABEL MAP ─────────────────────────────────────────────────────────────────
LABEL_MAP = {
    "ACTIVE":     "Active",
    "DEAD":       "Dead",
    "REPETITIVE": "Repetitive",
    "STAGNANT":   "Stagnant",
}

SYSTEM_PROMPT = (
    "You are a legal AI assistant specialising in Indian Supreme Court judgments. "
    "Read the judgment text and classify the case.\n\n"
    "Class definitions:\n"
    "- Active: Ongoing proceedings, pending appeals, remanded matters.\n"
    "- Dead: Final judgment, fully disposed, no pending appeal.\n"
    "- Stagnant: Adjourned sine die, no substantive progress.\n"
    "- Repetitive: Duplicate filings, same FIR, barred by res judicata.\n\n"
    "Format exactly as:\n"
    "CASE TYPE: <Active|Dead|Stagnant|Repetitive>\n"
    "CONFIDENCE: <high|medium|low>"
)

# ── PDF EXTRACTION ────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: Path, max_words: int = MAX_WORDS) -> str | None:
    """Extract and truncate text from a PDF. Returns None on failure."""
    try:
        text_parts = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
                if sum(len(p.split()) for p in text_parts) >= max_words:
                    break
        full_text = " ".join(text_parts)
        words = full_text.split()
        return " ".join(words[:max_words]) if words else None
    except Exception:
        return None


def find_pdf(row) -> Path | None:
    """Resolve PDF path from CSV row. Tries year subfolder first."""
    # Primary: PDF_ROOT / year / filename
    p = Path(PDF_ROOT) / str(row["folder"]) / row["file"]
    if p.exists():
        return p
    # Fallback: just filename under year folder (in case stored path differs)
    p2 = Path(PDF_ROOT) / str(row["folder"]) / Path(row["path"]).name
    if p2.exists():
        return p2
    return None


# ── WORKER ────────────────────────────────────────────────────────────────────
def process_row(row) -> dict | None:
    pdf_path = find_pdf(row)
    if pdf_path is None:
        return None

    text = extract_pdf_text(pdf_path)
    if not text or len(text.strip()) < MIN_CHARS:
        return None

    label = row["label_mapped"]
    conf  = str(row["confidence"]).lower()

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": text},
            {"role": "assistant", "content": f"CASE TYPE: {label}\nCONFIDENCE: {conf}"},
        ]
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Total rows       : {len(df)}")

    df_clean = df[df["label"] != "UNCLASSIFIED"].copy()
    df_clean["label_mapped"] = df_clean["label"].map(LABEL_MAP)
    print(f"  After drop UNCL  : {len(df_clean)}")
    print(f"  Class distribution:")
    for lbl, cnt in df_clean["label_mapped"].value_counts().items():
        print(f"    {lbl:<12} {cnt:>6}")

    # Make output directory if needed
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    rows = [row for _, row in df_clean.iterrows()]
    results = []
    skipped_notfound = 0
    skipped_notext   = 0

    print(f"\nExtracting PDFs with {MAX_WORKERS} threads...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, row): row for row in rows}
        with tqdm(total=len(rows), unit="pdf") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                else:
                    row = futures[future]
                    pdf_path = find_pdf(row)
                    if pdf_path is None:
                        skipped_notfound += 1
                    else:
                        skipped_notext += 1
                pbar.update(1)
                pbar.set_postfix(built=len(results), skip=skipped_notfound + skipped_notext)

    elapsed = time.time() - t0
    print(f"\n✅ Built       : {len(results)} examples")
    print(f"   Skipped (PDF not found) : {skipped_notfound}")
    print(f"   Skipped (text too short): {skipped_notext}")
    print(f"   Time          : {elapsed/60:.1f} min")

    # ── CLASS DISTRIBUTION IN OUTPUT ─────────────────────────────────────────
    from collections import Counter
    def get_label(ex):
        for line in ex["messages"][-1]["content"].split("\n"):
            if line.startswith("CASE TYPE:"):
                return line.split(":", 1)[1].strip()
        return "Unknown"

    dist = Counter(get_label(ex) for ex in results)
    print(f"\n   Output distribution:")
    for cls in ["Active", "Dead", "Repetitive", "Stagnant"]:
        print(f"    {cls:<12} {dist.get(cls, 0):>6}")

    # ── WRITE JSONL ───────────────────────────────────────────────────────────
    print(f"\nWriting {OUT_FILE} ...")
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for ex in results:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    size_mb = Path(OUT_FILE).stat().st_size / 1e6
    print(f"✅ Done — {Path(OUT_FILE).name}  ({size_mb:.1f} MB)")
    print(f"\nNext step: upload this file to Drive at:")
    print(f"  MyDrive/Project_2_Final/legal_training_data/dataset_from_csv.jsonl")
    print(f"  Colab Cell 1 will detect it and skip all PDF extraction automatically.")


if __name__ == "__main__":
    main()
