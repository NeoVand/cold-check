"""Zoom crops (600 dpi) of every panel of a figure, for the visual check: ./run.sh zoomall.py fig1 [fig2 ...]"""
import importlib, importlib.util, sys
sys.path.insert(0, ".")
import kit
BOX = {
  "fig1": {"a": (0, 0, 50, 50), "b": (50, 0, 116, 50), "c": (116, 0, 183, 50), "d": (0, 48, 92, 147), "e": (92, 48, 183, 147),
           "d_mid": (25, 75, 65, 115), "e_mid": (120, 75, 160, 115)},
  "fig3": {"a_small": (0, 0, 92, 72), "a_25": (120, 10, 183, 72), "b": (0, 76, 64, 145), "c": (64, 76, 130, 145),
           "d": (128, 76, 183, 145), "e": (0, 143, 183, 196)},
  "fig4": {"a_t1_top": (0, 0, 92, 40), "a_t1_bot": (0, 38, 92, 92), "a_t2": (90, 0, 183, 92), "b": (0, 88, 183, 160)},
  "fig5": {"a": (0, 0, 110, 90), "b": (105, 0, 183, 90)},
  "fig_arch": {"a_left": (0, 0, 64, 80), "a_block": (60, 8, 183, 80), "b": (0, 82, 78, 136), "c": (74, 82, 140, 136),
               "d": (138, 82, 183, 136)},
  "fig2": {"a": (0, 0, 122, 62), "b": (122, 0, 183, 62), "c": (0, 60, 122, 104), "d": (122, 60, 183, 104)},
}
for name in sys.argv[1:]:
    spec = importlib.util.spec_from_file_location(name, f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    fig = m.build() if not hasattr(m, "FIG") else m.FIG
    for p in kit.zoom(fig, name, BOX[name], dpi=600):
        print(p)
