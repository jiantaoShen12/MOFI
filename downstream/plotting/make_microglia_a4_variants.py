#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

A4_LANDSCAPE = (3508, 2480)  # 300 dpi
BG = (249, 248, 244)
PANEL_BG = (255, 255, 255)
TEXT = (24, 28, 33)
MUTED = (80, 86, 92)
ACCENT = (25, 102, 153)
BORDER = (210, 214, 220)
RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.ANTIALIAS)


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def panel_image(img: Image.Image, size: Tuple[int, int], *, pad: int = 24) -> Image.Image:
    canvas = Image.new("RGB", size, PANEL_BG)
    inner_w = max(10, size[0] - 2 * pad)
    inner_h = max(10, size[1] - 2 * pad)
    fitted = img.copy()
    fitted.thumbnail((inner_w, inner_h), RESAMPLE)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def combine_two(img1: Image.Image, img2: Image.Image, size: Tuple[int, int], *, title_left: str, title_right: str) -> Image.Image:
    canvas = Image.new("RGB", size, PANEL_BG)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, bold=True)
    gap = 26
    top = 56
    half_w = (size[0] - gap - 24 * 2) // 2
    inner_h = size[1] - top - 24
    p1 = panel_image(img1, (half_w, inner_h), pad=12)
    p2 = panel_image(img2, (half_w, inner_h), pad=12)
    x1 = 24
    x2 = 24 + half_w + gap
    canvas.paste(p1, (x1, top))
    canvas.paste(p2, (x2, top))
    draw.text((x1, 18), title_left, font=title_font, fill=MUTED)
    draw.text((x2, 18), title_right, font=title_font, fill=MUTED)
    return canvas


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    if hasattr(draw, "textlength"):
        return int(draw.textlength(text, font=font))
    return int(draw.textsize(text, font=font)[0])


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: Tuple[int, int], font: ImageFont.ImageFont, fill, max_width: int, line_gap: int = 8) -> int:
    x, y = xy
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        test = w if not cur else f"{cur} {w}"
        if text_width(draw, test, font) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def draw_round_box(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], *, radius: int, fill, outline=None, width: int = 1) -> None:
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    else:
        draw.rectangle(box, fill=fill, outline=outline)


def make_page(out_png: Path, title: str, subtitle: str, panels: List[Tuple[str, str, Image.Image]], supported: str, avoid: str, rank_note: str) -> None:
    canvas = Image.new("RGB", A4_LANDSCAPE, BG)
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(54, bold=True)
    subtitle_font = load_font(28)
    label_font = load_font(26, bold=True)
    caption_font = load_font(22)
    note_head_font = load_font(24, bold=True)
    note_font = load_font(21)

    draw.text((96, 72), title, font=title_font, fill=TEXT)
    draw.text((96, 146), subtitle, font=subtitle_font, fill=MUTED)
    draw_round_box(draw, (2710, 62, 3390, 186), radius=24, fill=(235, 242, 247), outline=None)
    draw.text((2740, 98), rank_note, font=load_font(28, bold=True), fill=ACCENT)

    left = 92
    top = 230
    gap_x = 44
    gap_y = 40
    panel_w = 1640
    panel_h = 860

    positions = [
        (left, top),
        (left + panel_w + gap_x, top),
        (left, top + panel_h + gap_y),
        (left + panel_w + gap_x, top + panel_h + gap_y),
    ]

    for (label, caption, img), (x, y) in zip(panels, positions):
        draw_round_box(draw, (x, y, x + panel_w, y + panel_h), radius=26, fill=PANEL_BG, outline=BORDER, width=3)
        draw.ellipse((x + 18, y + 18, x + 70, y + 70), fill=(230, 236, 242))
        tw = text_width(draw, label, label_font)
        draw.text((x + 44 - tw / 2, y + 24), label, font=label_font, fill=TEXT)
        draw.text((x + 92, y + 24), caption, font=caption_font, fill=TEXT)
        panel = panel_image(img, (panel_w - 36, panel_h - 90), pad=12)
        canvas.paste(panel, (x + 18, y + 72))

    note_y = 2048
    box_h = 300
    draw_round_box(draw, (92, note_y, 3416, note_y + box_h), radius=26, fill=(255, 252, 247), outline=BORDER, width=3)
    draw.text((122, note_y + 22), "Supported Biological Claim", font=note_head_font, fill=TEXT)
    draw.text((1710, note_y + 22), "Avoid Overclaim", font=note_head_font, fill=TEXT)
    draw_wrapped(draw, supported, (122, note_y + 62), note_font, MUTED, max_width=1460, line_gap=6)
    draw_wrapped(draw, avoid, (1710, note_y + 62), note_font, MUTED, max_width=1610, line_gap=6)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    canvas.save(out_png.with_suffix('.pdf'), resolution=300.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    figs = run_root / 'coarse_dense' / 'sample' / 'figures'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectory = open_rgb(figs / 'dense_time_dual_trajectory.png')
    fate = open_rgb(figs / 'cell_type_fate_dynamics.png')
    program = open_rgb(figs / 'program_dynamics.png')
    program_heatmap = open_rgb(figs / 'program_dynamics_heatmap.png')
    key_dot = open_rgb(figs / 'key_molecule_dotplot.png')
    key_volcano = open_rgb(figs / 'key_molecule_volcano.png')

    home_main = open_rgb(figs / 'tf_regulation_umap__Homeostatic/primary/regulation_umap/main_variable_umap.png')
    home_sub = open_rgb(figs / 'tf_regulation_umap__Homeostatic/primary/regulation_umap/sub_variable_umap.png')
    dam_main = open_rgb(figs / 'tf_regulation_umap__DAM_like/primary/regulation_umap/main_variable_umap.png')
    dam_sub = open_rgb(figs / 'tf_regulation_umap__DAM_like/primary/regulation_umap/sub_variable_umap.png')

    home_combo = combine_two(home_main, home_sub, (1600, 800), title_left='Main-space local geometry', title_right='Mapped protein geometry')
    dam_combo = combine_two(dam_main, dam_sub, (1600, 800), title_left='Main-space local geometry', title_right='Mapped protein geometry')

    make_page(
        out_dir / 'a4_v1_conservative_alignment.png',
        'Version 1 | Conservative Alignment Story',
        'Best for a reality-first manuscript panel: use only claims that remain aligned with observed time structure.',
        [
            ('A', 'Shared RNA-to-protein temporal backbone', trajectory),
            ('B', 'Observed versus predicted cell-state redistribution', fate),
            ('C', 'RNA program shifts across dense time', program),
            ('D', 'Path-specific key molecules', key_dot),
        ],
        supported='The main support is that the model preserves broad microglial state organization over time and recovers a dominant path with homeostatic protein features and reduced remodeling-associated RNA programs.',
        avoid='Do not present this page as proof of deterministic Homeostatic-to-DAM conversion or exact endpoint fate prediction. The fate panel is validation-level, not the central biology claim.',
        rank_note='Most Conservative\nHighest factual safety',
    )

    make_page(
        out_dir / 'a4_v2_molecular_axis.png',
        'Version 2 | Homeostatic-versus-Remodeling Molecular Axis',
        'Best for the strongest biology: the figure is centered on path-specific molecules rather than on fate endpoints.',
        [
            ('A', 'Shared temporal geometry across RNA and mapped protein space', trajectory),
            ('B', 'Stable key molecules along the dominant path', key_dot),
            ('C', 'Differential support for path-versus-background molecules', key_volcano),
            ('D', 'Program-level direction of change', program_heatmap),
        ],
        supported='This version supports a reference-state balance axis: P2RY12/CD54/CD49f/CD64-high protein readouts coexist with lower Ifit3/Isg15/Cst7/Lpl remodeling-associated RNA signals along the dominant path.',
        avoid='Do not describe the volcano panel as a strict late-versus-early endpoint comparison. It is a path-versus-background contrast and should be framed that way.',
        rank_note='Best Biology\nBest paper-facing narrative',
    )

    make_page(
        out_dir / 'a4_v3_local_regulatory_geometry.png',
        'Version 3 | Distinct Local Regulatory Neighborhoods',
        'Best for an insight-driven supplementary main-text figure: emphasize basin structure, not a single linear activation continuum.',
        [
            ('A', 'Global temporal manifold used as geometric context', trajectory),
            ('B', 'Homeostatic neighborhood in RNA and mapped protein geometry', home_combo),
            ('C', 'DAM-like neighborhood in RNA and mapped protein geometry', dam_combo),
            ('D', 'Molecular anchor for interpreting the neighborhoods', key_dot),
        ],
        supported='This version supports the idea that Homeostatic and DAM-like states occupy different local coupling neighborhoods. The biology is strongest when described as distinct basins within a shared reference manifold.',
        avoid='Do not claim that these UMAPs prove a causal regulatory circuit or a single monotonic progression. They are local geometry summaries and need the molecule panel to stay biologically anchored.',
        rank_note='Most Insightful\nMore interpretive, still defensible',
    )

    summary = out_dir / 'A4_VARIANTS_README.md'
    summary.write_text(
        '# A4 variants\n\n'
        '- `a4_v1_conservative_alignment.pdf`: safest, reality-first version.\n'
        '- `a4_v2_molecular_axis.pdf`: strongest biology and best main-text candidate.\n'
        '- `a4_v3_local_regulatory_geometry.pdf`: most insight-driven, but slightly more interpretive.\n',
        encoding='utf-8'
    )


if __name__ == '__main__':
    main()
