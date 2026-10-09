from __future__ import annotations

AUTHOR5_ORDER = ["RG", "IPC", "IN-fetal", "IN-CGE", "IN-MGE"]

AUTHOR5_COLORS = {
    "RG": "#B7D98C",
    "IPC": "#4E8B57",
    "IN-fetal": "#F3BCCB",
    "IN-CGE": "#D96C93",
    "IN-MGE": "#7B6FB2",
}

AUTHOR5_CELLTYPE_ALIAS = {
    "RG": "radial glial cell",
    "IPC": "neural progenitor cell",
    "IN-fetal": "inhibitory interneuron",
    "IN-CGE": "caudal ganglionic eminence derived interneuron",
    "IN-MGE": "medial ganglionic eminence derived interneuron",
}

CELLTYPE_TO_AUTHOR5 = {v: k for k, v in AUTHOR5_CELLTYPE_ALIAS.items()}


def author5_color(label: str, fallback: str = "#9AA4AF") -> str:
    key = str(label).strip()
    if key in AUTHOR5_COLORS:
        return AUTHOR5_COLORS[key]
    alias = CELLTYPE_TO_AUTHOR5.get(key)
    if alias is not None:
        return AUTHOR5_COLORS.get(alias, fallback)
    return fallback

