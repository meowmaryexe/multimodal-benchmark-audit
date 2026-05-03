import matplotlib.pyplot as plt
import numpy as np

models = ["Qwen2B", "Qwen7B", "InternVL2-4B"]
categories = ["Lookup", "Compositional", "Yes/No"]

image_color = "#4C72B0"
no_image_color = "#DDDDDD"

data = {
    "Qwen2B": {
        "image": [0.740, 0.375, 0.750],
        "no_image": [0.033, 0.082, 0.393],
        "image_ci": [(0.720, 0.759), (0.335, 0.416), (0.566, 0.873)],
        "no_image_ci": [(0.026, 0.042), (0.062, 0.109), (0.236, 0.576)],
    },
    "Qwen7B": {
        "image": [0.731, 0.491, 0.762],
        "no_image": [0.041, 0.202, 0.524],
        "image_ci": [(0.680, 0.777), (0.415, 0.567), (0.549, 0.894)],
        "no_image_ci": [(0.024, 0.069), (0.148, 0.271), (0.324, 0.717)],
    },
    "InternVL2-4B": {
        "image": [0.628, 0.400, 0.762],
        "no_image": [0.029, 0.124, 0.476],
        "image_ci": [(0.573, 0.680), (0.329, 0.475), (0.549, 0.894)],
        "no_image_ci": [(0.015, 0.054), (0.082, 0.181), (0.283, 0.676)],
    },
}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)

x = np.arange(len(categories))
width = 0.35

for ax, model in zip(axes, models):
    image_vals = np.array(data[model]["image"])
    no_image_vals = np.array(data[model]["no_image"])

    image_err = np.array([
        [val - lo for val, (lo, hi) in zip(image_vals, data[model]["image_ci"])],
        [hi - val for val, (lo, hi) in zip(image_vals, data[model]["image_ci"])],
    ])

    no_image_err = np.array([
        [val - lo for val, (lo, hi) in zip(no_image_vals, data[model]["no_image_ci"])],
        [hi - val for val, (lo, hi) in zip(no_image_vals, data[model]["no_image_ci"])],
    ])

    ax.bar(
        x - width / 2,
        image_vals,
        width,
        label="Image",
        yerr=image_err,
        capsize=2,
        error_kw=dict(linewidth=1),
        color=image_color,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.bar(
        x + width / 2,
        no_image_vals,
        width,
        label="No image",
        yerr=no_image_err,
        capsize=2,
        error_kw=dict(linewidth=1),
        color=no_image_color,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )

    ax.set_title(model, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="y", labelsize=10)

axes[0].set_ylabel("Accuracy", fontsize=11)

fig.suptitle("Visual reliance varies by task type", fontsize=12, y=0.98)
fig.legend(
    ["Image", "No image"],
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.93),
)

fig.tight_layout(rect=[0, 0, 1, 0.88])

plt.savefig("results/figures/main_visual_reliance.png", dpi=300, bbox_inches="tight")
plt.savefig("results/figures/main_visual_reliance.pdf", bbox_inches="tight")

print("Saved figure to results/figures/main_visual_reliance.png and .pdf")