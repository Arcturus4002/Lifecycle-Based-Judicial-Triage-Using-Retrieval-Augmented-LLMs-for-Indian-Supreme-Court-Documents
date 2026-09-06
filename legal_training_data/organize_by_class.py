"""
Organize validated case PDFs into per-class folders.

Reads a CSV produced by validation (the one with `your_decision` filled in)
and copies each case's PDF into a folder named after its FINAL label:

    <output_root>/Active/<files>.pdf
    <output_root>/Repetitive/<files>.pdf
    <output_root>/Dead/<files>.pdf

LABEL RESOLUTION (important):
    your_decision = "Confirm"          -> use rule_label
    your_decision = "Edit_to_Active"   -> label = Active
    your_decision = "Edit_to_Repetitive" -> label = Repetitive
    your_decision = "Edit_to_Dead"     -> label = Dead
    your_decision = "Reject"           -> SKIP this case (excluded)
    blank or other                     -> SKIP with warning

Files are COPIED (not moved) so your source corpus stays intact. Any case
where the source PDF is missing is logged and skipped, not silently dropped.

USAGE:
    python organize_by_class.py --input validation_set_confirmed.csv \\
        --output-root .\\dataset

    # Dry run — see what WOULD be copied, no files touched:
    python organize_by_class.py --input validation_set_confirmed.csv \\
        --output-root .\\dataset --dry-run

    # Move instead of copy (use with care; original files are gone after):
    python organize_by_class.py --input validation_set_confirmed.csv \\
        --output-root .\\dataset --move
"""

import argparse
import csv
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


VALID_FINAL_LABELS = {"Active", "Repetitive", "Dead"}


def resolve_final_label(decision: str, rule_label: str) -> tuple[str | None, str]:
    """
    Return (final_label, status) where status is one of:
        "ok"         — final_label is valid, copy the file
        "rejected"   — explicitly rejected by reviewer, skip silently
        "unmarked"   — no decision filled in, skip with warning
        "unknown"    — decision string we don't recognize
    """
    decision = (decision or "").strip()
    rule_label = (rule_label or "").strip()

    if not decision:
        return None, "unmarked"

    if decision.lower() == "reject":
        return None, "rejected"

    if decision.lower() == "confirm":
        if rule_label in VALID_FINAL_LABELS:
            return rule_label, "ok"
        return None, "unknown"

    # Edit_to_X form — accept any case
    if decision.lower().startswith("edit_to_"):
        suffix = decision.split("_", 2)[-1]
        # Normalize capitalization
        candidate = suffix.strip().capitalize()
        if candidate in VALID_FINAL_LABELS:
            return candidate, "ok"
        return None, "unknown"

    return None, "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="Input CSV with file_path, rule_label, your_decision columns")
    ap.add_argument("--output-root", required=True, type=Path,
                    help="Folder to create per-class subfolders under")
    ap.add_argument("--ignore-decisions", action="store_true",
                    help="Ignore the your_decision column entirely; use rule_label "
                         "for every row regardless of what's in your_decision. "
                         "Useful when you want to organize all candidates without "
                         "having validated them, or when your CSV doesn't have a "
                         "decision column at all.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would happen without copying any files")
    ap.add_argument("--move", action="store_true",
                    help="Move files instead of copying (DESTROYS originals)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite destination files if they already exist")
    args = ap.parse_args()

    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        return 1

    # Read the CSV
    rows = []
    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            log.error("Input CSV has no header row.")
            return 1
        required = {"file_path", "rule_label"}
        if not args.ignore_decisions:
            required.add("your_decision")
        missing = required - set(reader.fieldnames)
        if missing:
            log.error("Input CSV is missing required columns: %s", missing)
            log.error("Found columns: %s", reader.fieldnames)
            return 1
        rows = list(reader)

    log.info("Loaded %d rows from %s", len(rows), args.input)

    # First pass: classify every row
    plan: list[tuple[Path, str, str]] = []  # (src_pdf, final_label, status)
    statuses: Counter = Counter()
    label_distribution: Counter = Counter()

    if args.ignore_decisions:
        log.info("--ignore-decisions: using rule_label for every row.")

    for i, row in enumerate(rows, 1):
        rule_label = (row.get("rule_label") or "").strip()

        if args.ignore_decisions:
            if rule_label in VALID_FINAL_LABELS:
                final_label, status = rule_label, "ok"
            else:
                final_label, status = None, "unknown"
        else:
            decision = row.get("your_decision", "")
            final_label, status = resolve_final_label(decision, rule_label)
        statuses[status] += 1

        if status != "ok":
            if status == "unmarked":
                log.warning("Row %d: no decision; skipping (file=%s)",
                            i, row.get("case_filename", "?"))
            elif status == "unknown":
                if args.ignore_decisions:
                    log.warning("Row %d: rule_label %r is not a valid class; skipping",
                                i, rule_label)
                else:
                    log.warning("Row %d: unrecognized decision %r; skipping",
                                i, row.get("your_decision", ""))
            # rejected = silent skip
            continue

        src = Path(row.get("file_path", ""))
        if not src:
            log.warning("Row %d: empty file_path; skipping", i)
            statuses["missing_path"] += 1
            continue

        plan.append((src, final_label, status))
        label_distribution[final_label] += 1

    log.info("")
    log.info("Decision summary:")
    for status, n in statuses.most_common():
        log.info("  %-20s %5d", status, n)
    log.info("")
    log.info("Final label distribution (cases that will be copied):")
    for label, n in label_distribution.most_common():
        log.info("  %-12s %5d", label, n)
    log.info("  %-12s %5d", "TOTAL", sum(label_distribution.values()))

    if not plan:
        log.error("Nothing to copy. Check that --input has Confirm/Edit_to_* decisions filled in.")
        return 1

    # Second pass: actually copy
    if args.dry_run:
        log.info("")
        log.info("DRY RUN — showing first 5 planned operations:")
        for src, label, _ in plan[:5]:
            dst = args.output_root / label / src.name
            log.info("  %s -> %s", src.name, dst)
        log.info("(would copy %d files total)", len(plan))
        return 0

    # Create output folders
    for label in label_distribution:
        (args.output_root / label).mkdir(parents=True, exist_ok=True)
    log.info("Created folders under %s", args.output_root)

    copied = 0
    overwritten = 0
    missing_source = 0
    skipped_existing = 0
    errors = 0

    for src, label, _ in plan:
        if not src.exists():
            log.warning("Source PDF not found: %s", src)
            missing_source += 1
            continue

        dst = args.output_root / label / src.name
        if dst.exists():
            if args.overwrite:
                overwritten += 1
            else:
                skipped_existing += 1
                continue

        try:
            if args.move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            copied += 1
        except Exception as e:
            log.error("Failed to %s %s -> %s: %s",
                      "move" if args.move else "copy", src, dst, e)
            errors += 1

    log.info("")
    log.info("Done.")
    log.info("  Files %s:        %d", "moved" if args.move else "copied", copied)
    if overwritten:
        log.info("  Overwritten:        %d", overwritten)
    if skipped_existing:
        log.info("  Skipped (already exist, use --overwrite): %d", skipped_existing)
    if missing_source:
        log.warning("  Missing source PDFs: %d (these were in the CSV but not on disk)", missing_source)
    if errors:
        log.error("  Errors:             %d", errors)

    log.info("")
    log.info("Output structure:")
    for label in sorted(label_distribution):
        folder = args.output_root / label
        n_in_folder = len(list(folder.glob("*.[pP][dD][fF]")))
        log.info("  %s/  (%d PDFs)", folder, n_in_folder)

    return 0


if __name__ == "__main__":
    sys.exit(main())