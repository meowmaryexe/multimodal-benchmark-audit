import matplotlib.pyplot as plt
import numpy as np

categories = ["Lookup", "Compositional", "Yes/No"]
with_image = np.array([0.740, 0.375, 0.750])
no_image = np.array([0.033, 0.082, 0.393])
drops = with_image - no_image

x = np.arange(len(categories))
width = 0.34

plt.figure(figsize=(7.2, 4.2))

plt.bar(x - width / 2, with_image, width, label="With image")
plt.bar(x + width / 2, no_image, width, label="No image")

for i, drop in enumerate(drops):
    y = max(with_image[i], no_image[i]) + 0.04
    plt.text(
        x[i],
        y,
        f"drop: {drop:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.ylabel("Exact match accuracy")
plt.xlabel("Question type")
plt.title("Visual reliance differs sharply across ChartQA question types")
plt.xticks(x, categories)
plt.ylim(0, 1.0)
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig("results/figures/qwen_chartqa_visual_reliance.pdf")
plt.savefig("results/figures/qwen_chartqa_visual_reliance.png", dpi=300)

print("Saved figure to results/figures/")