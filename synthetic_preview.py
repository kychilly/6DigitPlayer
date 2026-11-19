"""Writes synthetic_preview.png so you can eyeball the training data generator.
    python synthetic_preview.py                                   # generic frames
    python synthetic_preview.py --skin "C:/.../osu!/Skins/<skin>"  # frames drawn with your skin
Crosses mark the labels: green circle, yellow slider head, magenta spinner, cyan slider ball."""
import argparse
import cv2
import numpy as np
from synthetic import SyntheticRenderer

ap = argparse.ArgumentParser()
ap.add_argument("--skin", default=None)
a = ap.parse_args()
if a.skin:
    from skin_synthetic import SkinRenderer
    r = SkinRenderer(a.skin)
else:
    r = SyntheticRenderer()
cols = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
tiles = []
for i in range(12):
    img, objs = r.render(np.random.default_rng(i))
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    for o in objs:
        cv2.drawMarker(big, (int(o.x * 2), int(o.y * 2)), cols[o.cls], cv2.MARKER_CROSS, 10, 1)
    tiles.append(big)
cv2.imwrite("synthetic_preview.png", np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 12, 3)]))
print("wrote synthetic_preview.png")
