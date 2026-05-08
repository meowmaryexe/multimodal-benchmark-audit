import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

models = [
    "Qwen2-VL-2B",
    "Qwen2-VL-7B",
    "Phi-3.5-Vision",
    "InternVL2-4B\n500-example subset",
]

# Full data order: Lookup, Compositional, Yes/No
data = {
    "Qwen2-VL-2B": {
        "image": [0.746, 0.404, 0.750],
        "no_image": [0.034, 0.075, 0.393],
        "image_ci": [(0.726, 0.765), (0.365, 0.443), (0.566, 0.873)],
        "no_image_ci": [(0.027, 0.043), (0.057, 0.099), (0.236, 0.576)],
    },
    "Qwen2-VL-7B": {
        "image": [0.812, 0.544, 0.786],
        "no_image": [0.050, 0.077, 0.500],
        "image_ci": [(0.794, 0.829), (0.505, 0.583), (0.605, 0.898)],
        "no_image_ci": [(0.041, 0.061), (0.058, 0.101), (0.326, 0.674)],
    },
    "Phi-3.5-Vision": {
        "image": [0.804, 0.516, 0.893],
        "no_image": [0.046, 0.065, 0.750],
        "image_ci": [(0.786, 0.822), (0.477, 0.556), (0.728, 0.963)],
        "no_image_ci": [(0.037, 0.056), (0.048, 0.088), (0.566, 0.873)],
    },
    "InternVL2-4B\n500-example subset": {
        "image": [0.628, 0.400, 0.762],
        "no_image": [0.029, 0.124, 0.476],
        "image_ci": [(0.573, 0.680), (0.329, 0.475), (0.549, 0.894)],
        "no_image_ci": [(0.015, 0.054), (0.082, 0.181), (0.283, 0.676)],
    },
}

image_color = "#4C72B0"
no_image_color = "#E0E0E0"


def make_grouped_bar_figure(
    categories,
    indices,
    output_stem,
    figsize=(12.8, 3.7),
    legend_y=1.03,
):
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)

    x = np.arange(len(categories))
    width = 0.30

    for ax, model in zip(axes, models):
        image_vals = np.array([data[model]["image"][i] for i in indices])
        no_image_vals = np.array([data[model]["no_image"][i] for i in indices])

        image_ci = [data[model]["image_ci"][i] for i in indices]
        no_image_ci = [data[model]["no_image_ci"][i] for i in indices]

        image_err = np.array([
            [val - lo for val, (lo, hi) in zip(image_vals, image_ci)],
            [hi - val for val, (lo, hi) in zip(image_vals, image_ci)],
        ])

        no_image_err = np.array([
            [val - lo for val, (lo, hi) in zip(no_image_vals, no_image_ci)],
            [hi - val for val, (lo, hi) in zip(no_image_vals, no_image_ci)],
        ])

        ax.bar(
            x - width / 2,
            image_vals,
            width,
            label="Image",
            yerr=image_err,
            capsize=2,
            error_kw={"linewidth": 0.9},
            color=image_color,
            edgecolor="black",
            linewidth=0.4,
        )

        ax.bar(
            x + width / 2,
            no_image_vals,
            width,
            label="No Image",
            yerr=no_image_err,
            capsize=2,
            error_kw={"linewidth": 0.9},
            color=no_image_color,
            edgecolor="black",
            linewidth=0.4,
        )

        ax.set_title(model, fontsize=10.5)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9.5)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.linspace(0, 1.0, 6))
        ax.grid(axis="y", alpha=0.18)
        ax.tick_params(axis="y", labelsize=9)

    axes[0].set_ylabel("Exact-match accuracy", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, legend_y),
        fontsize=10.5,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    png_path = os.path.join(OUTPUT_DIR, f"{output_stem}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{output_stem}.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


# Main-paper figure: stable categories only
make_grouped_bar_figure(
    categories=["Lookup", "Compositional"],
    indices=[0, 1],
    output_stem="main_visual_reliance",
    figsize=(12.8, 3.6),
    legend_y=1.04,
)

# Appendix figure: yes/no only, because estimates are noisier
make_grouped_bar_figure(
    categories=["Yes/No"],
    indices=[2],
    output_stem="appendix_yesno_visual_reliance",
    figsize=(12.8, 3.4),
    legend_y=1.05,
)
