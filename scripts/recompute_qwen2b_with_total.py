import csv
from collections import defaultdict
from pathlib import Path

INPUT_PATH = Path("results/qwen/qwen2b_subtype_results_full.csv")


def categorize_question_with_total(question):
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


stats = defaultdict(lambda: {"correct": 0, "total": 0})

with INPUT_PATH.open("r", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        category = categorize_question_with_total(row["question"])
        condition = row["condition"]
        correct = int(row["correct"])

        stats[(condition, category)]["total"] += 1
        stats[(condition, category)]["correct"] += correct

print("=== Qwen2B Recomputed WITH 'total' ===")

for category in ["lookup", "compositional", "yesno"]:
    print(f"\n{category}")

    for condition in ["image", "no_image"]:
        correct = stats[(condition, category)]["correct"]
        total = stats[(condition, category)]["total"]
        acc = correct / total if total else 0

        print(f"{condition}: {acc:.3f} ({correct}/{total})")