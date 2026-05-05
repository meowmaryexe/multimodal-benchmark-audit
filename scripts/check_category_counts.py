import csv
from collections import defaultdict
from pathlib import Path

FILES = {
    "qwen2b_full": "results/qwen/qwen2b_subtype_results_full.csv",
    "qwen7b_full": "results/qwen/qwen7b_chartqa_full.csv",
    "internvl_500": "results/internvl/internvl2_4b_chartqa_500.csv",
    "phi_500": "results/phi35/phi35_vision_chartqa_500.csv",
}

def categorize_question(question):
    q = question.lower()

    compositional_keywords = [
        "difference",
        "average",
        "ratio",
        "sum",
        "total",
        "how many more",
        "how many fewer",
        "greater than",
        "more than",
        "less than",
    ]

    if any(word in q for word in compositional_keywords):
        return "compositional"

    if q.startswith("is") or q.startswith("are") or q.startswith("does"):
        return "yesno"

    return "lookup"


for name, path_str in FILES.items():
    path = Path(path_str)

    print(f"\n=== {name} ===")
    print(f"path: {path}")

    if not path.exists():
        print("MISSING FILE")
        continue

    counts = defaultdict(int)
    condition_counts = defaultdict(lambda: defaultdict(int))

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            category = categorize_question(row["question"])
            condition = row["condition"]

            counts[category] += 1
            condition_counts[condition][category] += 1

    print("Total rows by recategorized label:")
    for cat in ["lookup", "compositional", "yesno"]:
        print(f"{cat}: {counts[cat]}")

    print("\nBy condition:")
    for condition in ["image", "no_image"]:
        print(f"{condition}:")
        for cat in ["lookup", "compositional", "yesno"]:
            print(f"  {cat}: {condition_counts[condition][cat]}")