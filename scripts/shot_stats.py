"""Analyze a globe screenshot: inner-disc color stats + horizontal/vertical detail."""
import sys
import numpy as np
from PIL import Image

path = sys.argv[1]
img = np.array(Image.open(path).convert("RGB")).astype(float)
h, w = img.shape[:2]
print("size", w, h)
cy, cx = h // 2, w // 2
yy, xx = np.mgrid[0:h, 0:w]
r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

disc = img[r < 120]
print("inner disc mean", disc.mean(axis=0).round(1))
print("inner disc std ", disc.std(axis=0).round(1))

g2 = img.mean(axis=2)
d0 = np.abs(np.diff(g2[cy, cx - 120 : cx + 120])).mean()
d1 = np.abs(np.diff(g2[cy - 120 : cy + 120, cx], axis=0)).mean()
print("detail h/v", round(float(d0), 2), round(float(d1), 2))

# quadrant means inside the disc to see land/ocean distribution
for name, mask in [
    ("top-left", (xx < cx) & (yy < cy) & (r < 120)),
    ("top-right", (xx >= cx) & (yy < cy) & (r < 120)),
    ("bot-left", (xx < cx) & (yy >= cy) & (r < 120)),
    ("bot-right", (xx >= cx) & (yy >= cy) & (r < 120)),
]:
    px = img[mask]
    print(f"{name:10s} mean", px.mean(axis=0).round(1), "std", px.std(axis=0).round(1))
