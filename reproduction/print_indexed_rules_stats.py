import argparse
import json
from pathlib import Path


def _normalize_rules(rules):
    if not isinstance(rules, list):
        return []
    if rules and isinstance(rules[0], dict):
        return [r["rule"] for r in rules if isinstance(r, dict) and "rule" in r]
    return rules


def load_indexed_rules(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "indexed_rules" in data:
        indexed_rules = data["indexed_rules"]
    elif isinstance(data, list):
        indexed_rules = data
    else:
        raise ValueError(
            "Expected a dict with 'indexed_rules' or a list of clusters in the JSON."
        )
    if not isinstance(indexed_rules, list):
        raise ValueError("Expected 'indexed_rules' to be a list of clusters.")
    return indexed_rules


def summarize(indexed_rules):
    per_cluster = []
    total_rules = 0
    for i, cluster in enumerate(indexed_rules, start=1):
        title = None
        if isinstance(cluster, dict):
            title = cluster.get("title") or cluster.get("summary")
            rules = _normalize_rules(cluster.get("rules", []))
        else:
            rules = []
        title = title or f"cluster_{i}"
        count = len(rules)
        per_cluster.append((title, count))
        total_rules += count
    return len(indexed_rules), total_rules, per_cluster


def main():
    parser = argparse.ArgumentParser(
        description="Summarize clusters and rule counts from an indexed_rules JSON."
    )
    parser.add_argument("input_path", type=Path, help="Path to indexed_rules_*.json")
    args = parser.parse_args()

    indexed_rules = load_indexed_rules(args.input_path)
    cluster_count, total_rules, per_cluster = summarize(indexed_rules)

    print(f"File: {args.input_path}")
    print(f"Clusters: {cluster_count}")
    print(f"Total rules: {total_rules}")
    print("")
    print("Per-cluster:")
    for idx, (title, count) in enumerate(per_cluster, start=1):
        print(f"{idx:>2}. {title} ({count} rules)")


if __name__ == "__main__":
    main()
