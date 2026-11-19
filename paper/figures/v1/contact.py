"""Contact sheets: out/contact_all.png (the five figures and the balance table) and candidates/contact_fig5.png."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
F = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 44)
FS = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 30)


def sheet(items, out, cols=2, width=2200, note=None):
    ims = [(t, Image.open(p).convert("RGB")) for t, p in items]
    colw = width // cols
    scaled = [(t, im.resize((colw - 60, int(im.height * (colw - 60) / im.width)))) for t, im in ims]
    rows = [scaled[i:i + cols] for i in range(0, len(scaled), cols)]
    H = sum(max(im.height for _, im in r) + 110 for r in rows) + (60 + 40 * len(note) if note else 0) + 40
    S = Image.new("RGB", (width, H), "white")
    d = ImageDraw.Draw(S)
    y = 30
    for r in rows:
        for k, (t, im) in enumerate(r):
            d.text((k * colw + 30, y), t, fill=(27, 42, 74), font=F)
            S.paste(im, (k * colw + 30, y + 70))
        y += max(im.height for _, im in r) + 110
    if note:
        for i, line in enumerate(note):
            d.text((30, y + 40 * i), line, fill=(27, 42, 74), font=FS)
    S.save(out)
    print("wrote", out)


O = HERE / "out"
bal = (O / "balance.txt").read_text().splitlines()
sheet([(f"Fig. {i}", O / f"fig{i}.png") for i in range(1, 6)], O / "contact_all.png", cols=2,
      note=["Rose and cobalt balance (check_balance.py, 300 dpi renders)"] + bal)
C = HERE / "candidates"
sheet([("Option 1: one plane, fit time against size (not chosen)", C / "fig5_option_scatter.png"),
       ("Option 2: two aligned row panels (chosen)", O / "fig5.png")], C / "contact_fig5.png", cols=2, width=2600)
sheet([("Map options for Fig. 1d,e: radial tree (chosen), spring layout, area-even radial", C / "fig1_maps.png")],
      C / "contact_fig1.png", cols=1, width=1800)
