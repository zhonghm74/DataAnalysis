"""
Shared matplotlib/seaborn configuration with Chinese font support.

Usage:
    import plot_config  # import before any plt/sns calls
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_CN_FONT = None
for candidate in ["WenQuanYi Micro Hei", "Droid Sans Fallback",
                   "Noto Sans CJK SC", "SimHei", "Source Han Sans SC"]:
    if any(f.name == candidate for f in fm.fontManager.ttflist):
        _CN_FONT = candidate
        break

if _CN_FONT:
    plt.rcParams.update({
        "font.sans-serif": [_CN_FONT, "DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,
    })

import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.05, rc={
    "font.sans-serif": [_CN_FONT, "DejaVu Sans", "Arial"] if _CN_FONT else ["DejaVu Sans"],
    "axes.unicode_minus": False,
})

plt.rcParams.update({"figure.max_open_warning": 0})
