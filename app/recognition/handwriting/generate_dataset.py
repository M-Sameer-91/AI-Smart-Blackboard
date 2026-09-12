"""Generate and validate a Windows-safe 62-class handwriting font dataset.

The legacy dataset used filenames that differed only by letter case (``A_0001``
versus ``a_0001``), which collide on Windows.  This generator keeps case in a
dedicated path component and writes new samples below ``generated/images``.
It never removes or writes to the legacy ``images`` directory.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CANVAS_SIZE = 128
RENDER_SCALE = 4
SAMPLES_PER_CLASS = 120
EXPECTED_CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
EXPECTED_FONTS = (
    "Caveat-Bold",
    "Caveat-Medium",
    "Caveat-Regular",
    "Caveat-SemiBold",
    "Caveat-VariableFont_wght",
    "DancingScript-Bold",
    "DancingScript-Medium",
    "DancingScript-Regular",
    "DancingScript-SemiBold",
    "DancingScript-VariableFont_wght",
    "Handlee-Regular",
    "IndieFlower-Regular",
)


def character_group(character: str) -> str:
    if "A" <= character <= "Z":
        return "upper"
    if "a" <= character <= "z":
        return "lower"
    return "digit"


def windows_normalized(path: Path | str) -> str:
    """Normalize a relative path as Windows resolves it, even off Windows."""
    return str(path).replace("/", "\\").lower()


def render_character(
    character: str,
    font: ImageFont.FreeTypeFont,
    font_name: str,
    sample_index: int,
) -> Image.Image:
    """Render one centered, antialiased grayscale character deterministically."""
    seed = f"{font_name}|{character}|{sample_index}"
    rng = random.Random(seed)
    high_size = CANVAS_SIZE * RENDER_SCALE
    # Small, deterministic rendering variation keeps the ten samples/font useful
    # while retaining generous padding on the 128px canvas.
    image = Image.new("L", (high_size, high_size), color=255)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), character, font=font)
    width, height = right - left, bottom - top
    jitter_x, jitter_y = rng.randint(-10, 10), rng.randint(-10, 10)
    x = (high_size - width) // 2 - left + jitter_x
    y = (high_size - height) // 2 - top + jitter_y
    draw.text((x, y), character, font=font, fill=0)
    return image.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)


def validate_dataset(dataset_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Validate structure, paths, labels, readability, and case separation."""
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = list(reader)

    errors: list[str] = []
    if headers != ["image_path", "label", "source", "font"]:
        errors.append(f"Unexpected manifest headers: {headers}")

    labels = Counter(row.get("label", "") for row in rows)
    paths = [row.get("image_path", "") for row in rows]
    normalized_paths = [windows_normalized(path) for path in paths]
    labels_by_path: dict[str, set[str]] = defaultdict(set)
    missing = []
    unreadable = []
    for row, normalized in zip(rows, normalized_paths):
        labels_by_path[normalized].add(row.get("label", ""))
        file_path = dataset_root / row.get("image_path", "")
        if not file_path.is_file():
            missing.append(row.get("image_path", ""))
            continue
        try:
            with Image.open(file_path) as image:
                image.verify()
        except Exception:
            unreadable.append(row.get("image_path", ""))

    png_paths = list((dataset_root / "generated" / "images").rglob("*.png"))
    duplicate_paths = len(paths) - len(set(paths))
    windows_collisions = len(normalized_paths) - len(set(normalized_paths))
    conflicting = sum(len(item_labels) > 1 for item_labels in labels_by_path.values())
    upper_lower_collisions = 0
    for upper, lower in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"):
        upper_rows = [row for row in rows if row["label"] == upper]
        lower_rows = [row for row in rows if row["label"] == lower]
        upper_paths = {windows_normalized(row["image_path"]) for row in upper_rows}
        lower_paths = {windows_normalized(row["image_path"]) for row in lower_rows}
        upper_lower_collisions += len(upper_paths & lower_paths)
        upper_by_font_and_name = {(row["font"], Path(row["image_path"]).name): row for row in upper_rows}
        for lower_row in lower_rows:
            corresponding_upper = upper_by_font_and_name.get(
                (lower_row["font"], Path(lower_row["image_path"]).name)
            )
            if corresponding_upper and os.path.samefile(
                dataset_root / corresponding_upper["image_path"],
                dataset_root / lower_row["image_path"],
            ):
                upper_lower_collisions += 1

    if len(rows) != len(EXPECTED_CLASSES) * SAMPLES_PER_CLASS:
        errors.append(f"Expected 7440 manifest rows, found {len(rows)}")
    if set(labels) != set(EXPECTED_CLASSES) or len(labels) != len(EXPECTED_CLASSES):
        errors.append("Manifest does not contain exactly the expected 62 classes")
    wrong_counts = {label: count for label, count in labels.items() if count != SAMPLES_PER_CLASS}
    if wrong_counts:
        errors.append(f"Unexpected per-class counts: {wrong_counts}")
    if len(png_paths) != len(rows):
        errors.append(f"Expected {len(rows)} physical PNG files, found {len(png_paths)}")
    if missing:
        errors.append(f"Missing manifest images: {len(missing)}")
    if duplicate_paths:
        errors.append(f"Duplicate manifest paths: {duplicate_paths}")
    if windows_collisions:
        errors.append(f"Windows-normalized path collisions: {windows_collisions}")
    if conflicting:
        errors.append(f"Conflicting labels for physical paths: {conflicting}")
    if upper_lower_collisions:
        errors.append(f"Upper/lower path collisions: {upper_lower_collisions}")
    if unreadable:
        errors.append(f"Unreadable PNG files: {len(unreadable)}")

    return {
        "manifest_rows": len(rows),
        "physical_png_files": len(png_paths),
        "classes": len(labels),
        "samples_per_class": min(labels.values(), default=0),
        "missing_files": len(missing),
        "duplicate_paths": duplicate_paths,
        "conflicting_labels": conflicting,
        "windows_normalized_path_collisions": windows_collisions,
        "upper_lower_collisions": upper_lower_collisions,
        "unreadable_png_files": len(unreadable),
        "uppercase_lowercase_separation": upper_lower_collisions == 0,
        "errors": errors,
        "passed": not errors,
    }


def print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("HANDWRITING DATASET INTEGRITY")
    print("=" * 60)
    print(f"Manifest rows: {report['manifest_rows']}")
    print(f"Physical PNG files: {report['physical_png_files']}")
    print(f"Classes: {report['classes']}")
    print(f"Samples per class: {report['samples_per_class']}")
    print(f"\nMissing files: {report['missing_files']}")
    print(f"Duplicate paths: {report['duplicate_paths']}")
    print(f"Conflicting labels: {report['conflicting_labels']}")
    print(f"Windows normalized path collisions: {report['windows_normalized_path_collisions']}")
    print(f"Upper/lower collisions: {report['upper_lower_collisions']}")
    print(f"Unreadable PNG files: {report['unreadable_png_files']}")
    print("Uppercase/lowercase separation: " + ("PASS" if report["uppercase_lowercase_separation"] else "FAIL"))
    if report["errors"]:
        print("\nValidation errors:")
        for error in report["errors"]:
            print(f"- {error}")
    print("\n" + "=" * 60)
    print("STATUS: " + ("PASS" if report["passed"] else "FAIL"))
    print("=" * 60)


def generate_dataset(dataset_root: Path) -> dict[str, Any]:
    """Generate to a clean tree, validate it, then publish the root manifest."""
    fonts_root = dataset_root / "fonts"
    staged_root = dataset_root / "generated"
    images_root = staged_root / "images"
    staged_manifest = staged_root / "manifest.csv"
    font_paths = {name: fonts_root / f"{name}.ttf" for name in EXPECTED_FONTS}
    missing_fonts = [str(path) for path in font_paths.values() if not path.is_file()]
    if missing_fonts:
        raise FileNotFoundError("Required font files are missing: " + ", ".join(missing_fonts))
    if staged_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing generated dataset: {staged_root}. "
            "Choose a new output root or inspect the existing generated dataset."
        )

    rows: list[dict[str, str]] = []
    for font_name, font_path in font_paths.items():
        font = ImageFont.truetype(str(font_path), 360)
        for character in EXPECTED_CLASSES:
            group = character_group(character)
            character_dir = images_root / font_name / group / character
            character_dir.mkdir(parents=True, exist_ok=False)
            for sample_index in range(1, 11):
                image_path = character_dir / f"{sample_index:04d}.png"
                render_character(character, font, font_name, sample_index).save(image_path, format="PNG")
                rows.append({
                    "image_path": image_path.relative_to(dataset_root).as_posix(),
                    "label": character,
                    "source": "font",
                    "font": font_name,
                })

    with staged_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "source", "font"])
        writer.writeheader()
        writer.writerows(rows)

    report = validate_dataset(dataset_root, staged_manifest)
    print_report(report)
    if not report["passed"]:
        raise RuntimeError("Generated dataset failed integrity validation; root manifest was not changed.")

    # Publish only after the new dataset validates.  Legacy images remain intact.
    (dataset_root / "manifest.csv").write_text(staged_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Windows-safe handwriting dataset")
    parser.add_argument("--dataset-root", type=Path, default=Path("app/data/datasets/handwriting"))
    parser.add_argument("--validate-only", action="store_true", help="Validate generated/manifest.csv without generating")
    args = parser.parse_args()
    if args.validate_only:
        report = validate_dataset(args.dataset_root, args.dataset_root / "generated" / "manifest.csv")
        print_report(report)
        if not report["passed"]:
            raise SystemExit(1)
        return
    generate_dataset(args.dataset_root)


if __name__ == "__main__":
    main()
