# compute_transition_buckets.py

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

FILES = {
    "Qwen2-VL-2B": "results/qwen/qwen2b_subtype_results_full.csv",
    "Qwen2-VL-7B": "results/qwen/qwen7b_chartqa_full.csv",
    "Phi-3.5-vision": "results/phi35/phi35_vision_chartqa_full.csv",
    "InternVL2-4B": "results/internvl/internvl2_4b_chartqa_500.csv",
}

OUTPUT_CSV = "transition_analysis_results.csv"


def categorize_question(question):
    q = str(question).lower().strip()

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


def compute_transition_stats(df, model_name):
    rows = []

    for category in ["lookup", "compositional", "yesno"]:
        cat_df = df[df["category"] == category]

        pivot = cat_df.pivot_table(
            index="index",
            columns="condition",
            values="correct",
            aggfunc="first",
        )

        if "image" not in pivot.columns or "no_image" not in pivot.columns:
            raise ValueError(
                f"{model_name} / {category} is missing image or no_image condition."
            )

        both_correct = ((pivot["image"] == 1) & (pivot["no_image"] == 1)).sum()
        image_only = ((pivot["image"] == 1) & (pivot["no_image"] == 0)).sum()
        no_image_only = ((pivot["image"] == 0) & (pivot["no_image"] == 1)).sum()
        both_wrong = ((pivot["image"] == 0) & (pivot["no_image"] == 0)).sum()

        total = len(pivot)
        image_correct_total = both_correct + image_only

        visual_dependence_rate = (
            image_only / image_correct_total
            if image_correct_total > 0
            else 0.0
        )

        rows.append({
            "model": model_name,
            "category": category,
            "total_examples": int(total),
            "both_correct": int(both_correct),
            "image_only_correct": int(image_only),
            "no_image_only_correct": int(no_image_only),
            "both_wrong": int(both_wrong),
            "visual_dependence_rate": round(float(visual_dependence_rate), 3),
        })

    return rows


all_rows = []

for model_name, filename in FILES.items():
    print(f"Loading {filename}")

    df = pd.read_csv(filename)

    required = {"index", "condition", "question", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")

    # Recompute categories from question text for every file so all models use
    # the exact same categorization rules.
    df["category"] = df["question"].apply(categorize_question)

    rows = compute_transition_stats(df, model_name)
    all_rows.extend(rows)

results_df = pd.DataFrame(all_rows)

print("\n=== Transition Analysis ===")
print(results_df)

results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved to {OUTPUT_CSV}")