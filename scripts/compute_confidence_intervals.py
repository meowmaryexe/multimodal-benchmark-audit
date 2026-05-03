import math

# Wilson confidence interval for binomial accuracy
def wilson_interval(correct, total, z=1.96):
    if total == 0:
        return None

    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom

    return center - margin, center + margin


results = {
    "Qwen2B_lookup_image": (1434, 1938),
    "Qwen2B_lookup_no_image": (64, 1938),
    "Qwen2B_comp_image": (200, 534),
    "Qwen2B_comp_no_image": (44, 534),
    "Qwen2B_yesno_image": (21, 28),
    "Qwen2B_yesno_no_image": (11, 28),

    "Qwen7B_lookup_image": (231, 316),
    "Qwen7B_lookup_no_image": (13, 316),
    "Qwen7B_comp_image": (80, 163),
    "Qwen7B_comp_no_image": (33, 163),
    "Qwen7B_yesno_image": (16, 21),
    "Qwen7B_yesno_no_image": (11, 21),

    "InternVL_lookup_image": (194, 309),
    "InternVL_lookup_no_image": (9, 309),
    "InternVL_comp_image": (68, 170),
    "InternVL_comp_no_image": (21, 170),
    "InternVL_yesno_image": (16, 21),
    "InternVL_yesno_no_image": (10, 21),
}

print("=== Wilson 95% Confidence Intervals ===\n")

for name, (correct, total) in results.items():
    acc = correct / total
    lo, hi = wilson_interval(correct, total)

    print(f"{name}")
    print(f"accuracy: {acc:.3f} ({correct}/{total})")
    print(f"95% CI: [{lo:.3f}, {hi:.3f}]")
    print()