"""
classify_all.py
===============
Classify all judgment PDFs in a folder using the rule-based classifier.
Generates a CSV report + prints summary stats + sample cases per class.

Usage:
    python classify_all.py "C:/path/to/judgments"
    python classify_all.py "C:/path/to/judgments" --output results.csv
    python classify_all.py "C:/path/to/judgments" --workers 8
"""

import os
import re
import csv
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

WORKERS = 8
OUTPUT_FILE = "classification_report.csv"


def _compile(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]

REPETITIVE_STRONG = _compile([
    r"res\s+judicata", r"same\s+cause\s+of\s+action",
    r"identical\s+(FIR|complaint|petition|matter|issue)",
    r"multiplicity\s+of\s+proceedings", r"duplicate\s+(filing|petition|complaint|case|FIR)",
    r"already\s+(been\s+)?adjudicated", r"same\s+FIR", r"same\s+subject\s*matter",
    r"order\s+II\s+rule\s+2", r"section\s+11\s+(of\s+)?(the\s+)?CPC",
    r"abuse\s+of\s+(the\s+)?process",
])
REPETITIVE_MODERATE = _compile([
    r"companion\s+(case|petition|matter|appeal)", r"connected\s+(case|petition|matter|appeal)",
    r"clubbed\s+(with|together)", r"consolidated\s+(with|hearing)",
    r"tagged\s+(with|along)", r"lead\s+(case|matter|petition)",
    r"common\s+(question|issue|subject)",
])
STAGNANT_STRONG = _compile([
    r"adjourned\s+sine\s+die", r"kept\s+pending\s+indefinitely",
    r"no\s+(further\s+)?progress", r"languishing", r"dormant", r"no\s+effective\s+hearing",
])
STAGNANT_MODERATE = _compile([
    r"none\s+(appeared|present)\s+for\s+(the\s+)?(petitioner|appellant|respondent)",
    r"(petitioner|appellant)\s+(has\s+)?(not\s+)?(appeared|shown\s+interest)",
    r"no\s+steps?\s+(taken|have\s+been\s+taken)",
    r"matter\s+(has\s+been|was)\s+adjourned\s+(repeatedly|several\s+times)",
    r"not\s+been\s+prosecuted", r"want\s+of\s+prosecution",
])
ACTIVE_STRONG = _compile([
    r"remand(ed)?\s+(back\s+)?(to|for)", r"remit(ted)?\s+(back\s+)?(to|for)",
    r"(sent|send)\s+back\s+(to|for)", r"matter\s+(is\s+)?remanded",
    r"fresh\s+(hearing|consideration|adjudication|decision|trial|inquiry|investigation)",
    r"de\s+novo\s+(trial|hearing|consideration)",
    r"direct(ed|s)?\s+(the\s+)?(trial\s+court|high\s+court|tribunal|lower\s+court|authority)\s+to",
    r"liberty\s+(is\s+)?(granted|given|reserved)\s+to",
    r"liberty\s+to\s+(file|apply|approach|move)",
    r"next\s+date\s+of\s+hearing", r"listed?\s+(for|on)\s+\d",
    r"posted\s+for\s+(hearing|further\s+hearing|orders)",
    r"part(ly|ial(ly)?)\s+(allowed|disposed)", r"partially\s+set\s+aside",
])
ACTIVE_MODERATE = _compile([
    r"pending\s+(appeal|application|petition|review|reference|complaint)",
    r"appeal\s+(is\s+)?(still\s+)?pending", r"subject\s+to\s+(the\s+)?outcome",
    r"further\s+(orders?|proceedings?|inquiry|investigation|hearing)",
    r"shall\s+(be\s+)?(heard|considered|decided|taken\s+up)",
    r"interim\s+(order|relief|stay|injunction)", r"status\s+quo",
    r"stay\s+(granted|continued|extended)", r"without\s+prejudice\s+to",
    r"at\s+liberty\s+to",
])
DEAD_STRONG = _compile([
    r"(appeal|petition|application|complaint|case|SLP|writ\s+petition)\s+(is\s+)?(hereby\s+)?(dismissed|rejected|disposed\s+of)",
    r"(conviction|sentence)\s+(is\s+)?(hereby\s+)?(confirmed|upheld|affirmed|maintained)",
    r"(acquittal|order)\s+(is\s+)?(hereby\s+)?(confirmed|upheld|affirmed|maintained)",
    r"(appeal|petition|SLP)\s+(is\s+)?(hereby\s+)?allowed",
    r"petition\s+stands?\s+(disposed|closed|rejected)",
    r"case\s+stands?\s+closed",
    r"no\s+merit\s+in\s+(this|the)\s+(appeal|petition|application|SLP)",
    r"we\s+find\s+no\s+(merit|substance|force|reason)",
    r"decree\s+(is\s+)?(hereby\s+)?(passed|granted|made)",
    r"suit\s+(is\s+)?(hereby\s+)?(decreed|dismissed)",
    r"finally\s+(disposed|decided|adjudicated|settled)",
    r"nothing\s+survives?\s+in\s+(this|the)",
])
DEAD_MODERATE = _compile([
    r"accordingly\s+(disposed|dismissed|rejected|allowed)",
    r"no\s+(further\s+)?orders?\s+(are\s+)?(necessary|needed|required|called\s+for)",
    r"disposed\s+of\s+in\s+(the\s+)?above\s+terms",
    r"impugned\s+(order|judgment|decree)\s+(is\s+)?(set\s+aside|upheld|affirmed|quashed)",
])


def classify_rules(text):
    t = " ".join(text.split())
    scores = {"REPETITIVE": 0, "STAGNANT": 0, "ACTIVE": 0, "DEAD": 0}
    for p in REPETITIVE_STRONG:
        if p.search(t): scores["REPETITIVE"] += 3
    for p in REPETITIVE_MODERATE:
        if p.search(t): scores["REPETITIVE"] += 1
    for p in STAGNANT_STRONG:
        if p.search(t): scores["STAGNANT"] += 3
    for p in STAGNANT_MODERATE:
        if p.search(t): scores["STAGNANT"] += 1
    for p in ACTIVE_STRONG:
        if p.search(t): scores["ACTIVE"] += 3
    for p in ACTIVE_MODERATE:
        if p.search(t): scores["ACTIVE"] += 1
    for p in DEAD_STRONG:
        if p.search(t): scores["DEAD"] += 3
    for p in DEAD_MODERATE:
        if p.search(t): scores["DEAD"] += 1

    if scores["REPETITIVE"] >= 3:
        conf = "high" if scores["REPETITIVE"] >= 6 else "medium"
        return "REPETITIVE", conf, scores
    if scores["STAGNANT"] >= 3 and scores["ACTIVE"] <= scores["STAGNANT"]:
        conf = "high" if scores["STAGNANT"] >= 6 else "medium"
        return "STAGNANT", conf, scores
    if scores["ACTIVE"] >= 3:
        active_strong = sum(1 for p in ACTIVE_STRONG if p.search(t))
        if active_strong >= 1:
            conf = "high" if scores["ACTIVE"] >= 6 else "medium"
            return "ACTIVE", conf, scores
        elif scores["ACTIVE"] > scores["DEAD"]:
            conf = "medium" if scores["ACTIVE"] >= 4 else "low"
            return "ACTIVE", conf, scores
    if scores["DEAD"] >= 3:
        conf = "high" if scores["DEAD"] >= 6 else "medium"
        return "DEAD", conf, scores
    return "UNCLASSIFIED", "low", scores


def process_one(pdf_path):
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    pt = page.extract_text()
                    if pt: text += pt + "\n"
                except: continue
    except: pass
    if len(text.strip()) < 50:
        try:
            from pypdf import PdfReader
            for page in PdfReader(pdf_path).pages:
                try:
                    pt = page.extract_text()
                    if pt: text += pt + "\n"
                except: continue
        except: pass

    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 50:
        return {
            "file": Path(pdf_path).name, "folder": Path(pdf_path).parent.name,
            "path": pdf_path, "label": "UNCLASSIFIED", "confidence": "low",
            "words": 0, "score_active": 0, "score_dead": 0,
            "score_stagnant": 0, "score_repetitive": 0,
        }

    label, conf, scores = classify_rules(text)
    return {
        "file": Path(pdf_path).name, "folder": Path(pdf_path).parent.name,
        "path": pdf_path, "label": label, "confidence": conf,
        "words": len(text.split()),
        "score_active": scores["ACTIVE"], "score_dead": scores["DEAD"],
        "score_stagnant": scores["STAGNANT"], "score_repetitive": scores["REPETITIVE"],
    }


def main():
    parser = argparse.ArgumentParser(description="Classify all judgment PDFs")
    parser.add_argument("folder", type=str, help="PDF folder (recursive)")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    folder = Path(args.folder)
    pdfs = sorted(set(str(p) for ext in ("*.pdf", "*.PDF") for p in folder.rglob(ext)))

    print(f"{'='*60}")
    print(f"  Legal Case Classifier")
    print(f"{'='*60}")
    print(f"  Folder:  {folder}")
    print(f"  PDFs:    {len(pdfs):,}")
    print(f"  Workers: {args.workers}")
    print(f"  Output:  {args.output}")
    print(f"{'='*60}\n")

    if not pdfs:
        print("No PDFs found")
        return

    results = []
    counts = {"ACTIVE": 0, "DEAD": 0, "STAGNANT": 0, "REPETITIVE": 0, "UNCLASSIFIED": 0}
    processed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, p): p for p in pdfs}
        for fut in as_completed(futures):
            processed += 1
            try:
                result = fut.result()
                results.append(result)
                counts[result["label"]] = counts.get(result["label"], 0) + 1
            except:
                results.append({
                    "file": Path(futures[fut]).name, "folder": Path(futures[fut]).parent.name,
                    "path": futures[fut], "label": "ERROR", "confidence": "",
                    "words": 0, "score_active": 0, "score_dead": 0,
                    "score_stagnant": 0, "score_repetitive": 0,
                })

            if processed % 500 == 0 or processed == len(pdfs):
                elapsed = time.time() - t0
                rate = processed / elapsed
                eta = (len(pdfs) - processed) / rate if rate > 0 else 0
                print(f"  [{processed:,}/{len(pdfs):,}] {rate:.1f}/s | ETA {eta/60:.0f}m | "
                      f"A:{counts['ACTIVE']} D:{counts['DEAD']} S:{counts['STAGNANT']} R:{counts['REPETITIVE']} U:{counts['UNCLASSIFIED']}")

    results.sort(key=lambda r: (r["label"], r["file"]))

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "folder", "label", "confidence", "words",
            "score_active", "score_dead", "score_stagnant", "score_repetitive", "path"
        ])
        writer.writeheader()
        writer.writerows(results)

    total_time = time.time() - t0
    total = len(results)

    print(f"\n{'='*60}")
    print(f"  CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(f"  Total cases: {total:,}")
    print(f"  Time: {total_time/60:.1f} min ({total/total_time:.0f} PDFs/s)\n")

    for label in ["ACTIVE", "DEAD", "STAGNANT", "REPETITIVE", "UNCLASSIFIED"]:
        n = counts.get(label, 0)
        pct = 100 * n / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {label:<15s} {n:>6,}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Saved to: {args.output}")

    print(f"\n{'='*60}")
    print(f"  SAMPLE CASES PER CLASS")
    print(f"{'='*60}")
    for label in ["ACTIVE", "DEAD", "STAGNANT", "REPETITIVE"]:
        cases = [r for r in results if r["label"] == label]
        print(f"\n  {label} ({len(cases):,} cases):")
        for c in cases[:5]:
            print(f"    {c['file'][:70]}")
            print(f"      conf: {c['confidence']}, {c['words']:,} words, "
                  f"scores: A={c['score_active']} D={c['score_dead']} S={c['score_stagnant']} R={c['score_repetitive']}")


if __name__ == "__main__":
    main()