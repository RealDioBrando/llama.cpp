#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def create_image(path: Path, width: int, height: int, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), "#f4f7fb")
    draw = ImageDraw.Draw(image)
    draw.rectangle([width // 10, height // 10, width * 9 // 10, height * 9 // 10], outline="#2b6cb0", width=8)
    draw.text((width // 20, height // 20), label, fill="#1a365d")
    image.save(path, format="JPEG", quality=90)


def create_mm_dataset(path: Path, image: Path, question: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    item = {"type": "image", "path": [str(image)], "question": question, "answer": "perf_only"}
    path.write_text(json.dumps(item) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "assets")
    args = parser.parse_args()
    create_image(args.output / "image_320x320.jpg", 320, 320, "perf input 320x320")
    create_image(args.output / "image_1280x720.jpg", 1280, 720, "perf input 1280x720")
    question = "Describe this image briefly."
    create_mm_dataset(args.output / "image_320x320_questions.jsonl", args.output / "image_320x320.jpg", question)
    create_mm_dataset(args.output / "image_1280x720_questions.jsonl", args.output / "image_1280x720.jpg", question)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
