# scripts/analyze_qwen7b_distractor.py

import pandas as pd

main = pd.read_csv("results/qwen/qwen7b_chartqa_full.csv")
dist = pd.read_csv("results/qwen/qwen7b_distractor_500.csv")

main_500 = main[main["index"] < 500].copy()

rows = []

for category in ["lookup", "compositional", "yesno"]:
    for condition in ["image", "no_image"]:
        sub = main_500[
            (main_500["category"] == category) &
            (main_500["condition"] == condition)
        ]
        rows.append({
            "category": category,
            "condition": condition,
            "correct": int(sub["correct"].sum()),
            "total": len(sub),
            "accuracy": sub["correct"].mean(),
        })

    sub = dist[dist["category"] == category]
    rows.append({
        "category": category,
        "condition": "distractor_image",
        "correct": int(sub["correct"].sum()),
        "total": len(sub),
        "accuracy": sub["correct"].mean(),
    })

out = pd.DataFrame(rows)
print(out)
out.to_csv("qwen7b_distractor_analysis.csv", index=False)
