"""Shared chart style + a registry so every figure carries its caption and justification."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, ACCENT, MUTED, BAD, GOOD, GOLD = "#1F2D33", "#064A56", "#9DAFB3", "#C2410C", "#2F7D4F", "#B7791F"
EQUIP_COLORS = {"Dry Van": ACCENT, "Reefer": "#3B82C4", "Flatbed": GOLD}
TRAIN_C, VALID_C = ACCENT, BAD

plt.rcParams.update({
    "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": "#E3EAEC", "grid.linewidth": 0.7,
    "legend.frameon": False, "figure.dpi": 160, "savefig.bbox": "tight",
})


class FigureRegistry:
    """Saves figures and records (section, title, what it shows, finding, decision)."""

    def __init__(self, out_dir: str | Path, section: str):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.section = section
        self.meta_path = self.out / "captions.json"
        self.items = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else []
        self.items = [i for i in self.items if i["section"] != section]

    def save(self, fig, name: str, title: str, shows: str, finding: str, decision: str = ""):
        path = self.out / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        self.items = [i for i in self.items if i["file"] != path.name]
        self.items.append({"section": self.section, "file": path.name, "title": title,
                           "shows": shows, "finding": finding, "decision": decision})
        self.meta_path.write_text(json.dumps(self.items, indent=2))
        print(f"  saved {path.name}")
