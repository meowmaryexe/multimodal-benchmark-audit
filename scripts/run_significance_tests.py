import csv
from collections import defaultdict
from pathlib import Path

from statsmodels.stats.proportion import proportions_ztest


FILES = {
    "Qwen2-VL-2B": "results/qwen/qwen2b_subtype_results_full.csv",
    "Qwen2-VL-7B": "results/qwen/qwen7b_chartqa_full.csv",
    "Phi-3.5-vision-instruct": "results/phi35/phi35_vision_chartqa_full.csv",
    "InternVL2-4B": "results/internvl/internvl2_4b_chartqa_500.csv",
}

CATEGORIES_TO_TEST = ["lookup", "compositional"]


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


def format_pvalue(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3g}"


results = []

for model, path_str in FILES.items():
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(f"Missing file for {model}: {path}")

    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Recompute category from question so all files use the same final rule.
            category = categorize_question(row["question"])
            condition = row["condition"]
            correct = int(row["correct"])

            stats[(condition, category)]["correct"] += correct
            stats[(condition, category)]["total"] += 1

    for category in CATEGORIES_TO_TEST:
        image_correct = stats[("image", category)]["correct"]
        image_total = stats[("image", category)]["total"]

        no_image_correct = stats[("no_image", category)]["correct"]
        no_image_total = stats[("no_image", category)]["total"]

        count = [image_correct, no_image_correct]
        nobs = [image_total, no_image_total]

        z_stat, p_value = proportions_ztest(count=count, nobs=nobs)

        image_acc = image_correct / image_total
        no_image_acc = no_image_correct / no_image_total
        delta = no_image_acc - image_acc

        results.append(
            {
                "model": model,
                "category": category,
                "image": f"{image_acc:.3f} ({image_correct}/{image_total})",
                "no_image": f"{no_image_acc:.3f} ({no_image_correct}/{no_image_total})",
                "delta": f"{delta:.3f}",
                "z": f"{z_stat:.3f}",
                "p": format_pvalue(p_value),
            }
        )


print("\n=== Two-proportion z-tests: image vs no_image ===")
print("Testing lookup and compositional categories only.\n")

for r in results:
    print(
        f"{r['model']} | {r['category']} | "
        f"image: {r['image']} | no_image: {r['no_image']} | "
        f"delta: {r['delta']} | z: {r['z']} | p: {r['p']}"
    )

print("\nLaTeX table rows:")
for r in results:
    category_name = "Lookup" if r["category"] == "lookup" else "Compositional"
    print(
        f"{r['model']} & {category_name} & {r['image']} & "
        f"{r['no_image']} & {r['delta']} & ${r['p']}$ \\\\"
    )
    