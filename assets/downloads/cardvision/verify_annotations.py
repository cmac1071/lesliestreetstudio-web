#!/usr/bin/env python3
"""
Draw bounding boxes from annotations.json onto exported images for visual verification.

Usage:
    python3 verify_annotations.py <export_folder>

Where <export_folder> is an unzipped CardVision export containing:
    annotations.json
    images/  (folder of JPEG files)

Outputs annotated images to a 'verify_output/' folder alongside the export.
Each image gets labeled bounding boxes drawn over it.
"""

import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Color palette for distinct box colors
COLORS = [
    (255, 50, 50),    # red
    (50, 200, 50),    # green
    (50, 100, 255),   # blue
    (255, 200, 0),    # yellow
    (255, 100, 200),  # pink
    (0, 220, 220),    # cyan
    (255, 140, 0),    # orange
    (180, 80, 255),   # purple
]

SUIT_NAMES = {"S": "Spd", "H": "Hrt", "D": "Dia", "C": "Clb"}
RANK_DISPLAY = {"T": "10"}


def label_display(code):
    """Convert 'TH' -> '10-Hrt', 'AS' -> 'A-Spd', etc."""
    if len(code) != 2:
        return code
    rank = RANK_DISPLAY.get(code[0], code[0])
    suit = SUIT_NAMES.get(code[1], code[1])
    return f"{rank}-{suit}"


def draw_annotations(image_path, annotations, output_path):
    """Draw bounding boxes and labels on an image."""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Try to get a reasonable font size based on image dimensions
    font_size = max(16, min(img.width, img.height) // 30)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    for i, ann in enumerate(annotations):
        color = COLORS[i % len(COLORS)]
        c = ann["coordinates"]
        label = ann["label"]

        # Create ML format: x,y is center; width,height are box dimensions
        left = c["x"] - c["width"] / 2
        top = c["y"] - c["height"] / 2
        right = c["x"] + c["width"] / 2
        bottom = c["y"] + c["height"] / 2

        # Draw box (3px outline for visibility)
        for offset in range(3):
            draw.rectangle(
                [left - offset, top - offset, right + offset, bottom + offset],
                outline=color,
            )

        # Draw label background + text
        display = label_display(label)
        bbox = draw.textbbox((0, 0), display, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        padding = 4

        label_x = left
        label_y = top - text_h - padding * 2 - 2
        if label_y < 0:
            label_y = bottom + 2  # put below if no room above

        draw.rectangle(
            [label_x, label_y, label_x + text_w + padding * 2, label_y + text_h + padding * 2],
            fill=color,
        )
        draw.text(
            (label_x + padding, label_y + padding),
            display,
            fill=(255, 255, 255),
            font=font,
        )

        # Draw crosshair at center point
        cx, cy = c["x"], c["y"]
        cross_size = 8
        draw.line([(cx - cross_size, cy), (cx + cross_size, cy)], fill=color, width=2)
        draw.line([(cx, cy - cross_size), (cx, cy + cross_size)], fill=color, width=2)

    img.save(output_path, "JPEG", quality=95)
    return len(annotations)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 verify_annotations.py <export_folder>")
        print("  export_folder: unzipped CardVision export with annotations.json and images/")
        return 1

    export_dir = Path(sys.argv[1])

    ann_path = export_dir / "annotations.json"
    if not ann_path.exists():
        print(f"Error: {ann_path} not found")
        return 1

    with open(ann_path) as f:
        data = json.load(f)

    output_dir = export_dir.parent / "verify_output"
    output_dir.mkdir(exist_ok=True)

    print(f"\nAnnotation Overlay Verification")
    print(f"{'=' * 50}")
    print(f"Source: {export_dir}")
    print(f"Output: {output_dir}")
    print(f"Entries: {len(data)}")
    print()

    success = 0
    for entry in data:
        ref = entry["imagefilename"]
        annotations = entry["annotation"]

        # Resolve image path (handles "images/UUID.jpg" or just "UUID.jpg")
        image_path = export_dir / ref
        if not image_path.exists():
            basename = Path(ref).name
            image_path = export_dir / "images" / basename
        if not image_path.exists():
            image_path = export_dir / Path(ref).name

        if not image_path.exists():
            print(f"  SKIP (not found): {ref}")
            continue

        # Get image dimensions for context
        img = Image.open(image_path)
        w, h = img.size
        img.close()

        out_name = Path(ref).stem + "_verified.jpg"
        out_path = output_dir / out_name
        count = draw_annotations(image_path, annotations, out_path)

        labels = ", ".join(label_display(a["label"]) for a in annotations)
        print(f"  {out_name}")
        print(f"    Image: {w}x{h}px | {count} boxes: {labels}")
        success += 1

    print(f"\n{success}/{len(data)} images processed -> {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
