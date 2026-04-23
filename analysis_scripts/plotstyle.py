"""Plotting style for figures"""

import matplotlib as mpl

def set_rcparams():

    mpl.rcParams.update({
        # --- Figure layout ---
        "figure.figsize": (3.25, 2.5),   # Single-column ACS width
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.transparent": False,

        # --- Fonts ---
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,                  # ACS single-column default
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        # --- Axes ---
        "axes.linewidth": 0.8,
        "axes.labelpad": 2,
        "axes.titlepad": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # --- Lines ---
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "lines.markeredgewidth": 0.5,

        # --- Ticks ---
        "xtick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.size": 3,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # --- Legend ---
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.borderpad": 0.2,

        # --- Colors ---
        "axes.prop_cycle": mpl.cycler(color=[
            "#1f77b4",  # blue
            "#d62728",  # red
            "#2ca02c",  # green
            "#9467bd",  # purple
            "#ff7f0e",  # orange
        ]),

        # --- PDF/PS output ---
        "pdf.fonttype": 42,   # Embed fonts as TrueType (ACS-friendly)
        "ps.fonttype": 42,
    })
