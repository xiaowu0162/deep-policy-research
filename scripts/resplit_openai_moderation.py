from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_policy_research.examples import ExampleRecord


CATEGORY_CONFIG = {
    "S": {
        "name": "Sexual",
        "slug": "sexual",
        "description": "Decide whether a message should be rejected for sexual content.",
        "positive_target": 10,
    },
    "H": {
        "name": "Hate",
        "slug": "hate",
        "description": "Decide whether a message should be rejected for hateful content.",
        "positive_target": 10,
    },
    "V": {
        "name": "Violence",
        "slug": "violence",
        "description": "Decide whether a message should be rejected for violent content.",
        "positive_target": 10,
    },
    "HR": {
        "name": "Harassment",
        "slug": "harassment",
        "description": "Decide whether a message should be rejected for harassment.",
        "positive_target": 10,
    },
    "SH": {
        "name": "Self-Harm",
        "slug": "self_harm",
        "description": "Decide whether a message should be rejected for self-harm content.",
        "positive_target": 10,
    },
    "S3": {
        "name": "Sexual/Minors",
        "slug": "sexual_minors",
        "description": "Decide whether a message should be rejected for sexual content involving minors.",
        "positive_target": 10,
    },
    "H2": {
        "name": "Hate/Threatening",
        "slug": "hate_threatening",
        "description": "Decide whether a message should be rejected for threatening hateful content.",
        "positive_target": 5,
    },
    "V2": {
        "name": "Violence/Graphic",
        "slug": "violence_graphic",
        "description": "Decide whether a message should be rejected for graphic violent content.",
        "positive_target": 5,
    },
}

LABEL_MAP = {
    0: "APPROVED",
    1: "REJECTED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resplit the OpenAI moderation data into train/validation/test JSONL files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "openai_moderation" / "category_dataframes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "openai_moderation_resplit",
    )
    parser.add_argument("--seed", type=int, default=20260309)
    return parser.parse_args()


def _group_rows(df: pd.DataFrame, seed_key: str) -> list[dict]:
    working = df.reset_index(drop=True).copy()
    working["_row_order"] = working.index
    groups = []
    for prompt, group in working.groupby("prompt", sort=False):
        labels = sorted({int(value) for value in group["label"].tolist()})
        if len(labels) != 1:
            raise ValueError(f"conflicting labels for duplicated prompt: {prompt[:80]!r}")
        groups.append(
            {
                "prompt": prompt,
                "label": labels[0],
                "rows": group.to_dict(orient="records"),
            }
        )
    groups.sort(key=lambda item: item["prompt"])
    random.Random(seed_key).shuffle(groups)
    return groups


def _select_groups(groups: list[dict], target_rows: int) -> tuple[list[dict], list[dict]]:
    if target_rows == 0:
        return [], list(groups)
    backpointers: dict[int, list[int]] = {0: []}
    for index, group in enumerate(groups):
        size = len(group["rows"])
        for subtotal, chosen in list(backpointers.items()):
            new_total = subtotal + size
            if new_total > target_rows or new_total in backpointers:
                continue
            backpointers[new_total] = chosen + [index]
    if target_rows not in backpointers:
        raise ValueError(f"unable to select {target_rows} rows from grouped data")
    chosen_indexes = set(backpointers[target_rows])
    selected = [group for index, group in enumerate(groups) if index in chosen_indexes]
    remaining = [group for index, group in enumerate(groups) if index not in chosen_indexes]
    return selected, remaining


def _flatten_rows(groups: list[dict]) -> list[dict]:
    rows = [row for group in groups for row in group["rows"]]
    rows.sort(key=lambda row: row["_row_order"])
    return rows


def _build_records(domain_slug: str, split: str, rows: list[dict]) -> list[dict]:
    records = []
    for index, row in enumerate(rows, start=1):
        record = ExampleRecord(
            id=f"{domain_slug}_{split}_{index:04d}",
            input=row["prompt"],
            label=LABEL_MAP[int(row["label"])],
            split=split,
            source_type="seed",
        )
        records.append(record.to_dict())
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _split_domain(df: pd.DataFrame, domain_code: str, seed: int) -> dict[str, list[dict]]:
    positive_target = CATEGORY_CONFIG[domain_code]["positive_target"]
    negative_target = 20 - positive_target
    splits = {}
    remaining_by_label = {}
    for label_value, target in ((1, positive_target), (0, negative_target)):
        label_df = df[df["label"] == label_value]
        groups = _group_rows(label_df, seed_key=f"{seed}:{domain_code}:{label_value}")
        train_groups, groups = _select_groups(groups, target)
        validation_groups, groups = _select_groups(groups, target)
        remaining_by_label[label_value] = {
            "train": train_groups,
            "validation": validation_groups,
            "test": groups,
        }
    for split in ("train", "validation", "test"):
        split_rows = _flatten_rows(remaining_by_label[1][split]) + _flatten_rows(remaining_by_label[0][split])
        split_rows.sort(key=lambda row: row["_row_order"])
        splits[split] = split_rows
    return splits


def _summarize(records: list[dict]) -> dict[str, int]:
    counter = Counter(record["label"] for record in records)
    return {
        "total": len(records),
        "APPROVED": counter.get("APPROVED", 0),
        "REJECTED": counter.get("REJECTED", 0),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "source_dataset": "data/openai_moderation/category_dataframes",
        "format": "jsonl",
        "seed": args.seed,
        "label_space": ["APPROVED", "REJECTED"],
        "tasks": [],
    }

    for domain_code, config in CATEGORY_CONFIG.items():
        source_path = args.input_dir / f"{domain_code}_df.parquet"
        df = pd.read_parquet(source_path)
        domain_dir = args.output_dir / config["slug"]
        domain_dir.mkdir()

        splits = _split_domain(df=df, domain_code=domain_code, seed=args.seed)
        stats = {}
        for split_name, split_rows in splits.items():
            records = _build_records(config["slug"], split_name, split_rows)
            _write_jsonl(domain_dir / f"{split_name}.jsonl", records)
            stats[split_name] = _summarize(records)

        manifest["tasks"].append(
            {
                "task_id": f"openai_moderation__{config['slug']}",
                "domain_name": config["name"],
                "description": config["description"],
                "train_path": f"{config['slug']}/train.jsonl",
                "validation_path": f"{config['slug']}/validation.jsonl",
                "test_path": f"{config['slug']}/test.jsonl",
                "positive_target_per_train_validation_split": config["positive_target"],
                "stats": stats,
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
