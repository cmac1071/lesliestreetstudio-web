#!/usr/bin/env python3
"""
Merge exported card detection datasets into an existing Create ML training set.

Run from the directory containing both the source export and target dataset folders.
The script prompts for folder names interactively.

Source: a folder containing one or more .zip exports from CardVision,
        or a single already-unzipped export folder (with images/ and annotations.json).

Target: your existing Create ML training dataset folder containing annotations.json
        and image files (card_00000.jpg, card_00001.jpg, etc.) in the same directory.

Images are renamed to continue the card_NNNNN sequence, guaranteeing no collisions.
"""

import collections
import json
import os
import shutil
import sys
import tempfile
import zipfile
from glob import glob
from pathlib import Path


VALID_LABELS = {
    f"{rank}{suit}"
    for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    for suit in ["S", "H", "D", "C"]
}


def find_next_card_number(target_dir):
    """Find the highest card_NNNNN.jpg number in the target directory and return next."""
    max_num = -1
    for f in os.listdir(target_dir):
        if f.startswith("card_") and f.lower().endswith(".jpg"):
            stem = f[5:].split(".")[0]
            try:
                num = int(stem)
                max_num = max(max_num, num)
            except ValueError:
                pass
    return max_num + 1


def load_annotations(path):
    """Load a Create ML annotations JSON file."""
    with open(path) as f:
        return json.load(f)


def save_annotations(path, data):
    """Write a Create ML annotations JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Wrote {len(data)} entries to {os.path.basename(path)}")


def resolve_image_path(source_dir, image_ref):
    """
    Resolve an image filename reference from annotations.json to an actual file path.
    Handles both flat ('UUID.jpg') and pathed ('images/UUID.jpg') references.
    """
    # Try as-is relative to source_dir
    candidate = os.path.join(source_dir, image_ref)
    if os.path.isfile(candidate):
        return candidate

    # Try just the basename in images/ subdirectory
    basename = os.path.basename(image_ref)
    candidate = os.path.join(source_dir, "images", basename)
    if os.path.isfile(candidate):
        return candidate

    # Try basename directly in source_dir
    candidate = os.path.join(source_dir, basename)
    if os.path.isfile(candidate):
        return candidate

    return None


def validate_annotations(entries, source_label):
    """Validate annotation entries and print warnings for issues."""
    warnings = []
    for entry in entries:
        if "annotation" not in entry or "imagefilename" not in entry:
            warnings.append(f"Malformed entry (missing keys): {entry}")
            continue
        for ann in entry["annotation"]:
            label = ann.get("label", "")
            if label not in VALID_LABELS:
                warnings.append(f"Invalid label '{label}' in {entry['imagefilename']}")
            coords = ann.get("coordinates", {})
            w = coords.get("width", 0)
            h = coords.get("height", 0)
            if w <= 0 or h <= 0:
                warnings.append(
                    f"Invalid bbox in {entry['imagefilename']}: "
                    f"width={w}, height={h}"
                )
    if warnings:
        print(f"\n  Warnings from {source_label}:")
        for w in warnings:
            print(f"    - {w}")
    return warnings


def process_single_export(export_dir, target_dir, start_number):
    """
    Process one unzipped export. Copies images into target_dir with sequential
    card_NNNNN.jpg naming. Returns (new_annotation_entries, next_number).
    """
    ann_path = os.path.join(export_dir, "annotations.json")
    if not os.path.isfile(ann_path):
        print(f"  Skipping (no annotations.json): {export_dir}")
        return [], start_number

    entries = load_annotations(ann_path)
    new_entries = []
    current = start_number

    for entry in entries:
        old_ref = entry["imagefilename"]
        image_path = resolve_image_path(export_dir, old_ref)

        if image_path is None:
            print(f"  Warning: image not found for '{old_ref}', skipping")
            continue

        new_filename = f"card_{current:05d}.jpg"
        dest = os.path.join(target_dir, new_filename)
        shutil.copy2(image_path, dest)

        new_entry = {
            "imagefilename": new_filename,
            "annotation": entry["annotation"],
        }
        new_entries.append(new_entry)
        current += 1

    return new_entries, current


def extract_and_process_zip(zip_path, target_dir, start_number):
    """Extract a .zip export to a temp dir, process it, clean up."""
    print(f"\n  Processing: {os.path.basename(zip_path)}")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp)
        except zipfile.BadZipFile:
            print(f"  Error: invalid zip file, skipping")
            return [], start_number

        # The zip may contain a top-level folder (e.g., CardVisionExport_20260417/)
        # or extract directly. Find where annotations.json lives.
        ann_file = None
        for root, dirs, files in os.walk(tmp):
            if "annotations.json" in files:
                ann_file = root
                break

        if ann_file is None:
            print(f"  Error: no annotations.json found in zip, skipping")
            return [], start_number

        return process_single_export(ann_file, target_dir, start_number)


def print_label_distribution(annotations):
    """Print a summary of label counts across all annotations."""
    counts = collections.Counter()
    for entry in annotations:
        for ann in entry.get("annotation", []):
            counts[ann.get("label", "?")] += 1

    if not counts:
        print("  (no annotations)")
        return

    # Print in a grid: 4 suits across, 13 ranks down
    ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    suits = ["S", "H", "D", "C"]
    symbols = {"S": "\u2660", "H": "\u2665", "D": "\u2666", "C": "\u2663"}

    print(f"  {'':>4}", end="")
    for s in suits:
        print(f"  {symbols[s]:>6}", end="")
    print()

    for r in ranks:
        display_rank = "10" if r == "T" else r
        print(f"  {display_rank:>4}", end="")
        for s in suits:
            label = f"{r}{s}"
            c = counts.get(label, 0)
            print(f"  {c:>6}", end="")
        print()

    total = sum(counts.values())
    print(f"\n  Total annotations: {total}")

    # Flag any missing labels (0 count)
    missing = [f"{r}{s}" for r in ranks for s in suits if counts.get(f"{r}{s}", 0) == 0]
    if missing:
        print(f"  Cards with zero annotations ({len(missing)}): {', '.join(missing)}")


def main():
    print()
    print("Card Detection Dataset Merge Tool")
    print("=" * 50)
    print()
    print("Both folders should be in the current directory:")
    print(f"  {os.getcwd()}")
    print()

    source = input("Source folder (exports to merge): ").strip()
    target = input("Target folder (existing training set): ").strip()
    print()

    # Validate target
    if not os.path.isdir(target):
        print(f"Error: target folder not found: {target}")
        return 1

    target_ann_path = os.path.join(target, "annotations.json")
    if not os.path.isfile(target_ann_path):
        print(f"Error: no annotations.json in target folder")
        return 1

    # Validate source
    if not os.path.exists(source):
        print(f"Error: source not found: {source}")
        return 1

    # Load existing
    existing = load_annotations(target_ann_path)
    start_number = find_next_card_number(target)
    print(f"Existing dataset: {len(existing)} images")
    print(f"Next card number: card_{start_number:05d}.jpg")

    # Back up existing annotations.json
    backup_path = target_ann_path + ".bak"
    shutil.copy2(target_ann_path, backup_path)
    print(f"Backed up annotations.json to {os.path.basename(backup_path)}")

    # Determine what's in source: zip files, or an unzipped export folder
    all_new_entries = []
    current_number = start_number

    if os.path.isdir(source):
        zips = sorted(glob(os.path.join(source, "*.zip")))
        has_annotations = os.path.isfile(os.path.join(source, "annotations.json"))

        if zips:
            print(f"\nFound {len(zips)} zip file(s) to merge:")
            for z in zips:
                entries, current_number = extract_and_process_zip(
                    z, target, current_number
                )
                if entries:
                    validate_annotations(entries, os.path.basename(z))
                    all_new_entries.extend(entries)
                    print(f"    Added {len(entries)} images")

        elif has_annotations:
            print(f"\nProcessing unzipped export folder: {source}")
            entries, current_number = process_single_export(
                source, target, current_number
            )
            if entries:
                validate_annotations(entries, source)
                all_new_entries.extend(entries)
                print(f"  Added {len(entries)} images")

        else:
            print(f"Error: source folder contains no .zip files and no annotations.json")
            return 1

    elif source.endswith(".zip") and os.path.isfile(source):
        entries, current_number = extract_and_process_zip(
            source, target, current_number
        )
        if entries:
            validate_annotations(entries, os.path.basename(source))
            all_new_entries.extend(entries)
            print(f"  Added {len(entries)} images")
    else:
        print(f"Error: source must be a folder or .zip file")
        return 1

    if not all_new_entries:
        print("\nNo new images to merge.")
        return 0

    # Merge and save
    merged = existing + all_new_entries
    save_annotations(target_ann_path, merged)

    # Summary
    print()
    print("=" * 50)
    print("MERGE COMPLETE")
    print("=" * 50)
    print(f"  Previously: {len(existing)} images")
    print(f"  Added:      {len(all_new_entries)} images")
    print(f"  Total:      {len(merged)} images")
    print(f"  Range:      card_{start_number:05d}.jpg - card_{current_number - 1:05d}.jpg")
    print()
    print("Label distribution (merged dataset):")
    print_label_distribution(merged)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
