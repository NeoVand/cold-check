"""Cobalt and rose balance per figure, measured on the 300 dpi render (out/figN.png).

Every pixel is fitted as a tint of each anchor colour on white, p = 1 - a (1 - c) in sRGB (matplotlib composites
alpha in sRGB, so pale fills, glows and anti-aliased edges of one colour all lie on that line), and given to the
anchor whose tint line it lies closest to, if the distance is under 0.06 and the tint a is at least 0.04. Anchors:
  rose    the rose of the kit and the Witnesses kit, its light and deep tones
  cobalt  the cobalt of both kits and its light tone
  other   navy ink, the soft and grey inks, the map's water, shore, roads, woods and trees, the SET card hues, frame
          and guide greys, the architecture figure's panel tint and grid grey (structure: counted, not reported as either);
          a pixel exactly one of these colours counts as structure before any tint-line fit
Reported per figure: the area of rose and cobalt pixels (mm^2), the same weighted by tint strength a ("ink", mm^2 of
full-strength colour), and rose's share of rose + cobalt for both. A pixel where two colours overlap (rose glow under
a grey road) fits no line well and is left out. `--masks` also writes out/balance_<fig>.png, each pixel painted by its
class, for checking the classifier by eye.

    ./run.sh check_balance.py fig1 fig2 [--masks]
"""
import pathlib
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"                                     # measures the staged renders (kit.OUT)
PX_MM2 = (25.4 / 300) ** 2

ANCHORS = {
    "rose": ["#D6336C", "#D1477A", "#F1A3BC", "#E9A2BC", "#A8325E", "#9E1F4B", "#B52A5C", "#E3678F"],
    "cobalt": ["#2557A7", "#1F4FD1", "#173C9E", "#9DB8E0", "#A3B7EE", "#4F7EC4", "#1B3F7A"],
    "other": ["#1B2A4A", "#1F2A44", "#4C5B78", "#8A94A6", "#7D8BA8", "#BAC5D9", "#E6ECF8", "#C3CEE5", "#7C90BC",
              "#F2F1EE", "#ECEAE5", "#BAC1CD", "#C9C4BA", "#CDD4E0", "#E2E7EF", "#F2F5FB", "#5B8FC7", "#55A27A",
              "#8C6BB8", "#7A5BA8", "#F2F5FA", "#C9CED8", "#6B7487"],
}


def rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def classify(img):
    """img: H x W x 3 in [0, 1]. Returns class index per pixel (-1 none, 0 rose, 1 cobalt, 2 other) and tint a."""
    q = 1.0 - img.reshape(-1, 3)                                  # distance from white
    best_d = np.full(len(q), np.inf)
    best_c = np.full(len(q), -1)
    best_a = np.zeros(len(q))
    for ci, fam in enumerate(("rose", "cobalt", "other")):
        for h in ANCHORS[fam]:
            v = 1.0 - rgb(h)
            a = np.clip(q @ v / (v @ v), 0, 1)
            d = np.linalg.norm(q - a[:, None] * v[None], axis=1)
            m = d < best_d
            best_d[m], best_c[m], best_a[m] = d[m], ci, a[m]
    blank = q.max(1) < 0.02
    # a pixel that is exactly a structural colour (within 0.012 in every channel) is structure, whatever tint line it
    # also fits: very pale structure (a panel tint) lies near the pale end of the cobalt line too
    flat = img.reshape(-1, 3)
    for h in ANCHORS["other"]:
        exact = np.abs(flat - rgb(h)[None]).max(1) < 0.012
        best_c[exact], best_d[exact], best_a[exact] = 2, 0.0, 1.0
    ok = (best_d < 0.06) & (best_a >= 0.04) & ~blank
    cls = np.where(ok, best_c, -1)
    cls[blank] = -2
    return cls.reshape(img.shape[:2]), best_a.reshape(img.shape[:2])


def measure(name, masks=False):
    im = np.asarray(Image.open(OUT / f"{name}.png").convert("RGB"), float) / 255
    cls, a = classify(im)
    area = {k: float((cls == i).sum()) * PX_MM2 for i, k in enumerate(("rose", "cobalt", "other"))}
    ink = {k: float(a[cls == i].sum()) * PX_MM2 for i, k in enumerate(("rose", "cobalt", "other"))}
    un = float((cls == -1).sum()) * PX_MM2
    if masks:
        pal = {-2: (255, 255, 255), -1: (250, 200, 60), 0: (214, 51, 108), 1: (37, 87, 167), 2: (200, 205, 215)}
        out = np.zeros(cls.shape + (3,), np.uint8)
        for k, c in pal.items():
            out[cls == k] = c
        Image.fromarray(out).save(OUT / f"balance_{name}.png")
    r = dict(name=name, rose_area=area["rose"], cobalt_area=area["cobalt"], rose_ink=ink["rose"],
             cobalt_ink=ink["cobalt"], other_area=area["other"], unclassified=un)
    r["rose_share_area"] = r["rose_area"] / max(r["rose_area"] + r["cobalt_area"], 1e-9)
    r["rose_share_ink"] = r["rose_ink"] / max(r["rose_ink"] + r["cobalt_ink"], 1e-9)
    return r


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or ["fig1", "fig2"]
    masks = "--masks" in sys.argv
    print(f"{'figure':8s} {'rose mm2':>9s} {'cobalt mm2':>11s} {'rose share':>11s} | {'rose ink':>9s} {'cobalt ink':>11s} "
          f"{'rose share':>11s} | {'unclassified mm2':>16s}")
    for n in names:
        r = measure(n, masks)
        print(f"{n:8s} {r['rose_area']:9.0f} {r['cobalt_area']:11.0f} {100 * r['rose_share_area']:10.0f}% | "
              f"{r['rose_ink']:9.0f} {r['cobalt_ink']:11.0f} {100 * r['rose_share_ink']:10.0f}% | {r['unclassified']:16.0f}")


if __name__ == "__main__":
    main()
