import os
import pandas as pd


INPUT_PATH = "results/plotqa/qwen7b_plotqa_test.csv"
OUTPUT_PATH = "results/plotqa/qwen7b_plotqa_summary.csv"


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Could not find {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print("\nColumns:")
    print(df.columns.tolist())

    required = ["chart_index", "qa_index", "condition", "category", "correct"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print("\nExpected columns missing:", missing)
        print("Detected columns:", df.columns.tolist())
        raise ValueError("Column mismatch. Rename columns in script to match your CSV.")

    # Basic sanity checks
    print("\n=== Sanity checks ===")
    print("Total rows:", len(df))
    print("Unique charts:", df["chart_index"].nunique())
    print("Conditions:", sorted(df["condition"].unique()))
    print("Categories:", sorted(df["category"].unique()))

    dupes = df.duplicated(
        subset=["chart_index", "qa_index", "condition"]
    ).sum()
    print("Duplicate chart/question/condition rows:", dupes)

    if dupes > 0:
        print("\nWARNING: duplicates found. Keeping first occurrence only.")
        df = df.drop_duplicates(
            subset=["chart_index", "qa_index", "condition"],
            keep="first",
        )

    # Accuracy table
    summary = (
        df.groupby(["category", "condition"])
        .agg(
            correct=("correct", "sum"),
            total=("correct", "count"),
        )
        .reset_index()
    )

    summary["accuracy"] = summary["correct"] / summary["total"]

    print("\n=== PlotQA accuracy breakdown ===")
    print(summary.to_string(index=False))

    # Pivot for paper-friendly table
    pivot = summary.pivot(
        index="category",
        columns="condition",
        values="accuracy",
    ).reset_index()

    if "image" in pivot.columns and "no_image" in pivot.columns:
        pivot["delta_no_image_minus_image"] = pivot["no_image"] - pivot["image"]
        pivot["absolute_drop"] = pivot["image"] - pivot["no_image"]

    print("\n=== Paper-friendly table ===")
    print(pivot.to_string(index=False))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved summary to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
