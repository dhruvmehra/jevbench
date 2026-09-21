import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import math

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
ACCENT, ACCENT_SOFT, GRAY_BAR, ROW_TINT = "#2a78d6", "#cde2fb", "#c9c8c2", "#eef4fc"
plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

rows = [
    # name, sub, sst2, ag, b77, p95_ms, cost_per_1k_b77, emphasis
    ("JEV",                 "typesafe/jev-1.13 · API",            95.4, 85.8, 76.4,  670, 0.08,  True),
    ("Claude Sonnet 5",     "anthropic · API",                    95.6, 89.6, 77.4, 2500, 6.43,  False),
    ("GPT-5-mini",          "openai · API",                       95.0, 80.2, 73.6, 2000, 0.38,  False),
    ("DistilBERT fine-tuned","10k labelled examples · local",     91.0, 91.0, 88.0,   13, 0.00,  False),
    ("Laya",                "convai · open weights · local",      92.0, 90.6, 38.2,  140, 0.00,  False),
    ("BART zero-shot",      "facebook/bart-large-mnli · local",   89.6, 76.4, 42.8, 3000, 0.00,  False),
]

W, H = 16, 9.6
fig = plt.figure(figsize=(W, H), dpi=150, facecolor=SURF)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

# Title block
ax.text(0.7, H-0.75, "JEV vs LLMs vs BERT vs Laya on text classification", fontsize=23, weight="bold", color=INK, va="center")
ax.text(0.7, H-1.25, "Accuracy on 500 held-out examples per dataset, p95 latency per request, and API cost. Same label descriptions for every zero-shot model.",
        fontsize=11.5, color=INK2, va="center")

# Column layout
x_name = 0.7
cols = [("SST-2", "2 labels · sentiment", 4.6), ("AG News", "4 labels · topic", 7.3), ("Banking77", "77 labels · intent", 10.0)]
bar_w = 1.9
x_lat = 13.9
x_cost = 15.3

y_head = H - 2.05
for title, sub, x in cols:
    ax.text(x, y_head, title, fontsize=12.5, weight="bold", color=INK, va="center")
    ax.text(x, y_head-0.32, sub, fontsize=9.5, color=MUTED, va="center")
ax.text(x_lat, y_head, "p95 latency", fontsize=12.5, weight="bold", color=INK, va="center", ha="center")
ax.text(x_lat, y_head-0.32, "per request", fontsize=9.5, color=MUTED, va="center", ha="center")
ax.text(x_cost, y_head, "$ / 1k", fontsize=12.5, weight="bold", color=INK, va="center", ha="right")
ax.text(x_cost, y_head-0.32, "Banking77", fontsize=9.5, color=MUTED, va="center", ha="right")
ax.text(x_name, y_head, "Classifier", fontsize=12.5, weight="bold", color=INK, va="center")
ax.plot([0.7, W-0.7], [y_head-0.62, y_head-0.62], color="#dddcd6", lw=1)

row_h = 0.95
y0 = y_head - 0.62 - row_h/2 - 0.05
for i, (name, sub, s, a, b, lat, cost, emph) in enumerate(rows):
    y = y0 - i*row_h
    if emph:
        ax.add_patch(FancyBboxPatch((0.45, y-row_h/2+0.06), W-0.9, row_h-0.12, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=ROW_TINT, ec="none", zorder=0))
    ax.text(x_name, y+0.14, name, fontsize=12.5, weight="bold" if emph else "medium", color=ACCENT if emph else INK, va="center")
    ax.text(x_name, y-0.2, sub, fontsize=9, color=MUTED, va="center")
    for (t, _, x), val in zip(cols, (s, a, b)):
        # inline bar, 0..100 -> bar_w
        ax.add_patch(Rectangle((x, y-0.13), bar_w, 0.26, fc="#efeeea", ec="none", zorder=1))
        ax.add_patch(FancyBboxPatch((x, y-0.13), bar_w*val/100, 0.26, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=ACCENT if emph else GRAY_BAR, ec="none", zorder=2))
        ax.text(x+bar_w+0.12, y, f"{val:.1f}", fontsize=12, weight="bold" if emph else "normal", color=INK, va="center")
    # latency: log scale bar 10ms..3000ms mapped to 1.6 width, centered column
    lw_max = 1.4
    frac = (math.log10(lat)-1)/(math.log10(3000)-1)
    xl = x_lat-0.7
    ax.add_patch(Rectangle((xl, y-0.13), lw_max, 0.26, fc="#efeeea", ec="none", zorder=1))
    ax.add_patch(FancyBboxPatch((xl, y-0.13), lw_max*frac, 0.26, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=ACCENT if emph else GRAY_BAR, ec="none", zorder=2))
    lat_txt = f"{lat/1000:.1f} s" if lat >= 1000 else f"{lat} ms"
    ax.text(x_lat, y-0.33, lat_txt, fontsize=9.5, color=INK2, va="center", ha="center")
    cost_txt = "$0" if cost == 0 else f"${cost:.2f}"
    ax.text(x_cost, y, cost_txt, fontsize=12, weight="bold" if emph else "normal", color=INK, va="center", ha="right")
    if i < len(rows)-1:
        ax.plot([0.7, W-0.7], [y-row_h/2, y-row_h/2], color="#ecebe6", lw=0.8, zorder=0)

# Footer
yf = 0.62
ax.text(0.7, yf+0.32, "Latency measured with strictly sequential requests through OpenRouter (JEV, LLMs) or on an Apple-silicon Mac (local models); latency bar is log-scaled.",
        fontsize=9.3, color=MUTED, va="center")
ax.text(0.7, yf, "Fine-tuned DistilBERT saw 10k labelled training examples per dataset; every other model saw only label descriptions. ±2.5 pts at n=500. Local models: $0 API cost.",
        fontsize=9.3, color=MUTED, va="center")
fig.savefig("docs/assets/benchmark-table.png", dpi=150, facecolor=SURF)
print("saved")
