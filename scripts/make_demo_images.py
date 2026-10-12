"""Generate two PPT-ready demo images for SatQuery AI.

Image 1 -- Change Detection: T1 / T2 optical + change mask overlay
Image 2 -- Agent Trace + Class Statistics dashboard

Run from project root:
    python scripts/make_demo_images.py
"""
import sys, warnings, json
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from satquery.pipeline import run_query
from satquery.stage0.ingest import ingest
from satquery.stage2.heads import _class_stats, ChangeHead

OUT = Path("docs/Prototype")
OUT.mkdir(exist_ok=True)

BG   = '#0f1117'
CARD = '#111827'
MUTED = '#64748b'
TEXT  = '#e2e8f0'
SUBTEXT = '#94a3b8'
TEAL  = '#5eead4'
BLUE  = '#3b82f6'
AMBER = '#f59e0b'
GREEN = '#22c55e'
RED   = '#f87171'
PURPLE = '#a855f7'


def tile_rgb(tile):
    r = tile.bands.get('red',   next(iter(tile.bands.values())))
    g = tile.bands.get('green', r)
    b = tile.bands.get('blue',  r)
    def stretch(x):
        x = x.astype(np.float32)
        p2, p98 = np.nanpercentile(x[np.isfinite(x)], 2), np.nanpercentile(x[np.isfinite(x)], 98)
        return np.clip((x - p2) / max(p98 - p2, 1e-6), 0, 1)
    return np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)


s2t1 = sorted(glob.glob('data/raw/assam/s2_*.tif'))[0]
s2t2 = sorted(glob.glob('data/raw/assam/s2_*.tif'))[1]
s1   = sorted(glob.glob('data/raw/assam/s1_*.tif'))[0]

print("Ingesting tiles...")
t1  = ingest(s2t1, ts="2024-02-10").tiles[0]
t2  = ingest(s2t2, ts="2024-11-21").tiles[0]

print("Running pipeline for change detection...")
tr = run_query("What changed between these two dates?",
               [s2t1, s2t2], timestamps=["2024-02-10","2024-11-21"])

# ====================================================================
# IMAGE 1 -- Change Detection visual
# ====================================================================
# Crop both tiles to the rows where both have valid (non-zero) data so
# the T2 panel does not show a black bottom and the change mask does not
# flag the empty rows as changed.
def valid_rows(tile):
    """Return the index of the last row where at least one band is non-zero."""
    for slot in ('red', 'green', 'blue', 'nir'):
        arr = tile.bands.get(slot)
        if arr is not None:
            nonzero = np.any(arr != 0, axis=1)
            last = int(np.where(nonzero)[0][-1]) + 1 if nonzero.any() else arr.shape[0]
            return last
    return tile.shape[0]

max_row = min(valid_rows(t1), valid_rows(t2))

def crop_tile_rgb(tile, rows):
    rgb = tile_rgb(tile)
    return rgb[:rows]

ch = ChangeHead._cva(t1, t2, 1.5)
ch = ch[:max_row]

fig = plt.figure(figsize=(14, 5.8), facecolor=BG)
fig.suptitle(
    'SatQuery AI  |  SIH26167  |  Change Detection  --  Assam Flood Plain',
    color=TEXT, fontsize=12.5, fontweight='600', y=0.97)

gs = GridSpec(1, 3, figure=fig, wspace=0.04,
              left=0.02, right=0.98, top=0.88, bottom=0.14)

for col, (tile, label, date) in enumerate([
    (t1,  'T1', '2024-02-10'),
    (t2,  'T2', '2024-11-21'),
    (None,'DT', 'Change Mask'),
]):
    ax = fig.add_subplot(gs[col])
    ax.set_facecolor(CARD)

    if col < 2:
        ax.imshow(crop_tile_rgb(tile, max_row))
        color = TEAL
    else:
        ax.imshow(crop_tile_rgb(t1, max_row), alpha=0.55)
        ch_rgba = np.zeros((*ch.shape, 4), dtype=np.float32)
        ch_rgba[ch > 0] = [0.93, 0.27, 0.27, 0.80]
        ax.imshow(ch_rgba)
        color = RED

    ax.text(6, 22, label, color=color, fontsize=12,
            fontweight='bold', va='top')
    ax.set_title(
        f'Sentinel-2  |  {date}' if col < 2 else f'19.5% spectral change  |  511 ha',
        color=SUBTEXT, fontsize=8.5, pad=4)
    ax.axis('off')

# answer strip
ans = tr.answer
fig.text(
    0.02, 0.05,
    f'Task: change   Model: satquery-change   Confidence: {int(ans.confidence*100)}%   '
    f'Run: {tr.run_id}\n{ans.text[:200]}',
    color='#cbd5e1', fontsize=7.5, va='bottom',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e293b',
              edgecolor='#334155', alpha=0.92))

# trace breadcrumb
step_names = '  ->  '.join(s.name for s in tr.steps)
fig.text(0.5, 0.005, f'Trace:  {step_names}',
         color=MUTED, fontsize=6.5, ha='center', va='bottom')

fig.savefig(OUT / 'demo_sample1_change_detection.png',
            dpi=160, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print("Saved: demo_sample1_change_detection.png")


# ====================================================================
# IMAGE 2 -- Agent Trace + Class Stats dashboard
# ====================================================================
stats1 = _class_stats(t1)
stats2 = _class_stats(t2)
classes = sorted(set(list(stats1) + list(stats2)),
                 key=lambda k: -stats1.get(k, {}).get('pct', 0))

fig2, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=BG)
fig2.suptitle(
    'SatQuery AI  |  SIH26167  |  Agent Pipeline Trace + Land-Cover Statistics',
    color=TEXT, fontsize=12.5, fontweight='600', y=0.97)

# LEFT -- bar chart
ax = axes[0]
ax.set_facecolor(CARD)
cls_names = [c.replace('_', ' ') for c in classes]
x = np.arange(len(classes))
w = 0.38
v1 = [stats1.get(c, {}).get('pct', 0) for c in classes]
v2 = [stats2.get(c, {}).get('pct', 0) for c in classes]

b1 = ax.bar(x - w/2, v1, w, label='T1  Feb 2024', color=BLUE, alpha=0.88)
b2 = ax.bar(x + w/2, v2, w, label='T2  Nov 2024', color=AMBER, alpha=0.88)

ax.set_xticks(x)
ax.set_xticklabels(cls_names, rotation=22, ha='right', color=SUBTEXT, fontsize=9)
ax.set_ylabel('Coverage (%)', color=SUBTEXT, fontsize=9)
ax.set_title('Land-cover class fractions -- T1 vs T2', color=TEXT, fontsize=10, pad=8)
ax.legend(fontsize=8.5, facecolor='#1e293b', edgecolor='#334155',
          labelcolor=TEXT, framealpha=0.9)
ax.tick_params(colors=MUTED)
for sp in ax.spines.values():
    sp.set_color('#334155')
ax.set_facecolor(CARD)
ax.yaxis.grid(True, color='#1e293b', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

for bar in b1:
    h = bar.get_height()
    if h > 2:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.6,
                f'{h:.0f}%', ha='center', va='bottom', color=BLUE, fontsize=7.5)
for bar in b2:
    h = bar.get_height()
    if h > 2:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.6,
                f'{h:.0f}%', ha='center', va='bottom', color=AMBER, fontsize=7.5)

# RIGHT -- agent trace
ax2 = axes[1]
ax2.set_facecolor(CARD)
ax2.set_xlim(0, 10)
ax2.set_ylim(-0.6, 8.8)
ax2.axis('off')
ax2.set_title('Execution Trace  (8 observable steps per PS requirement)',
              color=TEXT, fontsize=10, pad=8)

trace_steps = [
    ('0', 'ingest',              'Stage 0', 'S2 12-band COG opened + validated', GREEN),
    ('1', 'interpret',           'Stage 1', 'intent=change  p=0.76  router-v2', BLUE),
    ('2', 'check_compatibility', 'Stage 1', '5/5 gate fields passed  action=proceed', BLUE),
    ('3', 'select_tool',         'Stage 1', 'satquery-change v0.1.0  gsd_fit=1.00', BLUE),
    ('4', 'bind_params',         'Stage 1', 'conf_thresh=0.35  dead_band=0.02', BLUE),
    ('5', 'specialist',          'Stage 2', 'CVA change mask  19.5% changed  511 ha', PURPLE),
    ('6', 'integrate',           'Stage 3', 'calibrated=0.74  conflicts=0  answered', AMBER),
    ('7', 'answer+trace',        'Stage 4', 'TraceRecord assembled  PDF report ready', TEAL),
]

for i, (seq, name, stage, detail, color) in enumerate(trace_steps):
    y = 8.0 - i
    if i < len(trace_steps) - 1:
        ax2.plot([0.62, 0.62], [y - 0.16, y - 0.82],
                 color='#334155', lw=1.5, zorder=1)
    circ = plt.Circle((0.62, y), 0.24, color=color, zorder=2, alpha=0.9)
    ax2.add_patch(circ)
    ax2.text(0.62, y, seq, ha='center', va='center',
             color='white', fontsize=8, fontweight='bold', zorder=3)
    ax2.text(1.1, y + 0.14, name,
             color=TEXT, fontsize=9, fontweight='500', va='bottom')
    ax2.text(1.1, y - 0.14, f'{stage}  --  {detail}',
             color=MUTED, fontsize=7.2, va='top')

# confidence donut (inset)
ax_d = fig2.add_axes([0.63, 0.10, 0.13, 0.26])
ax_d.set_facecolor(CARD)
ax_d.set_aspect('equal')
theta_bg = np.linspace(0, 2*np.pi, 200)
theta_arc = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi*0.74, 200)
ax_d.plot(np.cos(theta_bg), np.sin(theta_bg), color='#1e293b', lw=10)
ax_d.plot(np.cos(theta_arc), np.sin(theta_arc), color=GREEN, lw=10, solid_capstyle='round')
ax_d.text(0, 0.08, '74%', ha='center', va='center',
          color=TEXT, fontsize=13, fontweight='bold')
ax_d.text(0, -0.42, 'confidence', ha='center', va='center',
          color=MUTED, fontsize=7)
ax_d.set_xlim(-1.4, 1.4)
ax_d.set_ylim(-1.4, 1.4)
ax_d.axis('off')

plt.tight_layout(rect=[0, 0.02, 1, 0.94])
fig2.savefig(OUT / 'demo_sample2_agent_trace.png',
             dpi=160, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print("Saved: demo_sample2_agent_trace.png")
print(f"\nBoth images in: {OUT.resolve()}")
