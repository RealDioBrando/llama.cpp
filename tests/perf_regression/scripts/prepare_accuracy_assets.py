#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path


REPO_ID = "maoxx241/textvqa_subset"


def http_get(url: str, output: Path | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "llama-perf-regression"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
    return data


def repo_url(endpoint: str, path: str) -> str:
    return f"{endpoint.rstrip('/')}/datasets/{REPO_ID}/resolve/main/{path.lstrip('/')}"


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def unique_rows(rows: list[dict], count: int | None) -> list[dict]:
    seen = set()
    selected = []
    for row in rows:
        image = row["image"]
        if image in seen:
            continue
        seen.add(image)
        selected.append(row)
        if count is not None and len(selected) >= count:
            break
    return selected


def write_annotations(source: Path, output: Path, question_ids: set[int]) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    annotations = [
        item for item in data.get("annotations", [])
        if int(item.get("question_id", -1)) in question_ids
    ]
    output.write_text(
        json.dumps({"annotations": annotations}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_textvqa(count: int, full_dataset: bool, output_dir: Path, endpoint: str,
                    force: bool) -> None:
    metadata_dir = output_dir / "textvqa_json"
    images_dir = output_dir / "val_images"
    rows_path = metadata_dir / "textvqa_val.jsonl"
    annotations_path = metadata_dir / "textvqa_val_annotations.json"

    if force or not rows_path.exists():
        http_get(repo_url(endpoint, "textvqa_val.jsonl"), rows_path)
    if force or not annotations_path.exists():
        http_get(repo_url(endpoint, "textvqa_val_annotations.json"), annotations_path)

    all_rows = load_rows(rows_path)
    selected_rows = all_rows if full_dataset else unique_rows(all_rows, count)
    if not selected_rows:
        raise RuntimeError("TextVQA metadata contains no rows")

    for row in selected_rows:
        image_name = Path(row["image"]).name
        image_path = images_dir / image_name
        if force or not image_path.exists():
            http_get(repo_url(endpoint, f"train_images/{image_name}"), image_path)
        repo_root = Path(__file__).resolve().parents[1]
        try:
            row["image"] = image_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            row["image"] = str(image_path.resolve())

    stem = "textvqa_val_full" if full_dataset else f"textvqa_val_{len(selected_rows)}"
    subset_path = metadata_dir / f"{stem}.jsonl"
    subset_annotations_path = metadata_dir / f"{stem}_annotations.json"
    subset_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected_rows),
        encoding="utf-8",
    )
    question_ids = {int(row["question_id"]) for row in selected_rows}
    write_annotations(annotations_path, subset_annotations_path, question_ids)

    print(f"images={len({Path(row['image']).name for row in selected_rows})}")
    print(f"rows={len(selected_rows)}")
    print(f"dataset={subset_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the TextVQA accuracy subset.")
    parser.add_argument("--num-images", type=int, default=30)
    parser.add_argument("--full", action="store_true", help="prepare all TextVQA val rows")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "textvqa",
    )
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.full and args.num_images > 0:
        parser.error("--full and --num-images are mutually exclusive")
    if args.num_images < 0:
        parser.error("--num-images must be non-negative")

    count = None if args.full else args.num_images
    endpoint = args.hf_endpoint or "https://huggingface.co"
    try:
        prepare_textvqa(count, args.full, args.output_dir, endpoint, args.force)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
