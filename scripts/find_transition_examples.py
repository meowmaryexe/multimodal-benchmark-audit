# scripts/find_transition_examples.py

import os
import pandas as pd
from datasets import load_dataset

FILE = "results/qwen/qwen2b_subtype_results_full.csv"
IMAGE_OUT_DIR = "figures/qualitative_examples/source_images"

os.makedirs(IMAGE_OUT_DIR, exist_ok=True)

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


def assign_transition(row):
    if row["image"] == 1 and row["no_image"] == 1:
        return "both_correct"
    if row["image"] == 1 and row["no_image"] == 0:
        return "image_only"
    if row["image"] == 0 and row["no_image"] == 1:
        return "no_image_only"
    return "both_wrong"


print("Loading results...")
df = pd.read_csv(FILE)

print("Columns:", list(df.columns))

required = {"index", "condition", "question", "gold", "pred", "correct"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["category"] = df["question"].apply(categorize_question)

pivot = df.pivot_table(
    index="index",
    columns="condition",
    values="correct",
    aggfunc="first",
).reset_index()

image_rows = (
    df[df["condition"] == "image"]
    [["index", "category", "question", "gold", "pred"]]
    .rename(columns={"pred": "image_pred"})
)

no_image_rows = (
    df[df["condition"] == "no_image"]
    [["index", "pred"]]
    .rename(columns={"pred": "no_image_pred"})
)

merged = image_rows.merge(no_image_rows, on="index")
merged = merged.merge(pivot, on="index")

merged["transition"] = merged.apply(assign_transition, axis=1)

targets = [
    ("lookup", "image_only"),
    ("lookup", "both_wrong"),
    ("compositional", "image_only"),
    ("compositional", "both_wrong"),
    ("yesno", "both_correct"),
    ("yesno", "no_image_only"),
]

print("Loading ChartQA dataset for source images...")
dataset = load_dataset("lmms-lab/ChartQA", split="test")

chosen = []

for category, transition in targets:
    subset = merged[
        (merged["category"] == category) &
        (merged["transition"] == transition)
    ].copy()

    print("\n==============================")
    print(category, transition)
    print("==============================")

    if len(subset) == 0:
        print("NONE FOUND")
        continue

    # Print first 5 candidates so you can choose a good, readable one.
    for _, row in subset.head(5).iterrows():
        idx = int(row["index"])
        image_path = os.path.join(
            IMAGE_OUT_DIR,
            f"{category}_{transition}_{idx}.png"
        )

        dataset[idx]["image"].save(image_path)

        print(f"\nINDEX: {idx}")
        print(f"IMAGE SAVED: {image_path}")
        print(f"QUESTION: {row['question']}")
        print(f"GOLD: {row['gold']}")
        print(f"IMAGE PRED: {row['image_pred']}")
        print(f"NO IMAGE PRED: {row['no_image_pred']}")

    chosen.append(subset.head(5))

all_candidates = pd.concat(chosen, ignore_index=True)
all_candidates.to_csv(
    "qualitative_example_candidates.csv",
    index=False
)

print("\nSaved candidate table to qualitative_example_candidates.csv")
print(f"Saved source images to {IMAGE_OUT_DIR}")