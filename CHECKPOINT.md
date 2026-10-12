# SatQuery AI — Build Checkpoint
**SIH26167 · ISRO / Space Technology**
Last updated: 2026-09-25 (Benchmark coverage extended: BigEarthNet fuse
per-class IoU + LEVIR-CD change-mask F1/IoU folded into the PDF (Tables 8–9);
offline packaging done (vendored JS/fonts + int8 ONNX + torch-free path);
RSVQA-HR full download in progress; SAR gap fix scripted, awaiting Earthdata login)

**2026-10-12 update:** full-project audit + fix pass. Added `requirements.txt`,
`README.md`, `.gitignore`, `LICENSE` and a `tests/` suite; reconciled and
regenerated both report PDFs (benchmark §4 now reads JSON live; the completion
report gained an "Extended evaluation" section and dropped the stale EuroSAT
"held-out" framing); reworded the pitch docs for honesty (cross-signal
reconciliation, actual §9 model stack, 6 of 10 sites carry SAR). The CURRENT
supervised headline is VRSBench **83.84%** / RSVQA-LR 80.83% / CDVQA 77.18%
(training pass 3, `data/bench_metrics.json`) — the pass-2 table in the
2026-09-12 entry below (82.04 best) is historical. Still open: SAR backfill
(needs an Earthdata login), RSVQA-HR `test_phili`, LEVIR-CD SECOND.

---

## Session update (2026-09-25)

### BigEarthNet held-out fuse per-class IoU — DONE ✅ (`scripts/eval_ben_fuse_iou.py`)
No download (BEN patches already on disk). Held-out bench split (1,020 patches
with both S2+S1), multi-label per-class IoU over the 19 BEN-19 classes,
reproducing the deployed optical (caption_head) + SAR (s1_head) heads. Late
fusion (mean/max) vs each modality alone. **BOUNDED finding:** at the deployed
0.5 threshold max-fusion wins (macro-mIoU .322 vs optical .295), but with
per-class thresholds tuned on train, a well-operated optical-only head wins
(.351); SAR helps only radar-favourable classes (arable/urban/industrial/
water) and hurts spectral forest classes. → `data/ben_fuse_iou_metrics.json`,
benchmark PDF Table 8. Re-run: `python scripts/eval_ben_fuse_iou.py`.

### LEVIR-CD change-mask F1/IoU — DONE ✅ (`scripts/eval_levir_cd.py`)
Test-only ~70 MB HF download (`ericyu/LEVIRCD_Cropped_256`, 2,048 tiles).
Scores the deployed ChangeHead's unsupervised CVA mask against building-change
ground truth; `cva_norm` self-checked byte-identical to `ChangeHead._cva`.
**Honest lower bound:** deployed F1 0.085 / IoU 0.044 (best-F1 sweep 0.096);
precision pinned at the ~5% change-pixel base rate — CVA localises generic
spectral change, not buildings, on RGB. → `data/levir_cd_metrics.json`,
benchmark PDF Table 9. Re-run: `python scripts/eval_levir_cd.py`.

### Offline packaging — DONE ✅
- `docs/vendor/three.min.js` + `docs/vendor/fonts.css`/`fonts/*.woff2` (Inter,
  Geist Mono) vendored into `docs/index.html` (local-first, CDN/online
  fallback kept). Tile cache already warm.
- `scripts/export_onnx.py` → `models/onnx/`: int8 ONNX for opt_feat, sar_feat,
  caption_head, s1_head (194→49 MB; static QDQ for conv graphs, dynamic for
  heads; torch↔ONNX parity ~4e-7, int8 cosine ≥0.995).
- `satquery/stage2/onnx_infer.py`: torch-free (onnxruntime+numpy) offline LULC
  path, verified vs the deployed torch heads (optical max|Δp|≈0.04, top-1
  matches). Kept separate from the live torch backend.

### RSVQA-HR — DONE ✅ (`scripts/eval_rsvqa_hr.py`)
Full high-res split from Zenodo record 6344367 (Images.tar 13.5 GB → 10,659
tiles). Test set 1, 222,684 questions: **zero-shot 41.0% / supervised 85.6%**
(presence .90, comp .82, count .86, area .84 supervised). Count/area binned
into the paper's 5 ranges (strip the `m2` unit before parsing area digits);
supervised capped at 40k train examples for the 6 GB GPU (`--limit-train`),
zero-shot persisted before adaptation so an OOM can't lose it. → benchmark PDF
Section 10 / Table 10. Re-run: `python scripts/eval_rsvqa_hr.py --mode both`.

### SAR data gap — SCRIPTED, AWAITING EARTHDATA LOGIN ⏳ (`scripts/fetch_sar_asf.py`)
ASF OPERA RTC-S1 (analysis-ready) covers all 4 gap sites (gangotri/rondonia/
creekfire/nile — RTC scene within 1–4 days of each S2 date, confirmed via
`--dry-run`). Only blocker: an Earthdata login (set `EDL_TOKEN`, or
`EARTHDATA_USERNAME`/`PASSWORD`, or `~/.netrc`), then `python
scripts/fetch_sar_asf.py`. Writes the same s1 tile format as fetch_data and
updates the manifest; fuse then stops degrading to optical-only there.

---

## Session update (2026-09-24)

### Calibration fitting COMPLETE ✅ (`scripts/fit_calibration.py`)
Per-task temperature scaling (`p_cal = sigmoid(logit(p)/T)`, the exact
`integrate.calibrate` transform) fit on a held-out 50/50 split of RSVQA-LR +
VRSBench + CDVQA and evaluated on the disjoint half; ECE / MCE / NLL / Brier +
reliability tables + a confidence→correctness AUROC diagnostic land in
`data/calibration_metrics.json`. Two scorers measured:

1. **Zero-shot (deployed joint scorer)** — confidence is near-uninformative on
   foreign benchmarks: AUROC ≈ 0.5 (RSVQA 0.59, VRSBench 0.53, CDVQA 0.48),
   ECE 0.38–0.45. A single scalar T can only lower ECE by collapsing every
   confidence toward the base rate, so the fit **pins at the search ceiling**.
   Honest headline: a 98%-confident zero-shot answer is right only ~47% of the
   time — which is exactly why the pipeline grounds answer confidence in
   auditable spectral fractions + cross-model reconciliation, not raw softmax.
2. **Adapted (task-fit scorer = the mission-corpus deployment setting)** —
   confidence IS informative (AUROC 0.77–0.90); temperature scaling roughly
   halves ECE at a sensible T: vqa pooled T=2.11 (ECE .063→.020), CDVQA
   change-qa T=1.18 (ECE .051→.029).

`integrate.DEFAULT_TEMPERATURES` now ships the fitted values for the two tasks
the adapted scorer drives (`vqa`=2.11, `change-qa`=1.18); describe/ground/
change/fuse remain conservative **priors** (hand-designed bounded confidences,
no cached held-out set to fit — not fabricated). Quick eval re-run after the
change: router v1+v2 100%, pipeline 100%, abstention sane. Folded into
`docs/SatQuery_AI_Benchmark_Results.pdf` (Confidence calibration, Table 6).
Re-run: `python scripts/fit_calibration.py`.

### VRSBench caption + grounding COMPLETE ✅ (`scripts/eval_vrsbench_caption_grounding.py`)
Closes the caption-metrics + grounding-IoU coverage gap. Offline metrics
(`scripts/vlm_metrics.py`: BLEU-1..4, ROUGE-L, CIDEr-D, box IoU — self-checked,
no pycocoevalcap/nltk). Deployed heads run faithfully via the benchmark-PNG
ingest path. Both VRSBench sub-tasks are **out of distribution** and reported
as an honest lower bound → `data/vrsbench_caption_grounding_metrics.json`:
- **Caption** (full val, n=9350): BLEU-4 0.000, ROUGE-L 0.059, CIDEr-D 0.002.
  The CaptionHead is a BEN-19 land-cover describer (~27 words), VRSBench wants
  ~50-word scene prose — a genre mismatch, not a bug.
- **Grounding**: 0/16159 objects groundable. The system grounds 6 areal
  spectral classes; VRSBench grounds 26 discrete man-made object classes
  (vehicle/ship/airplane/…) — zero vocabulary overlap, so referring-object
  IoU is N/A by construction (stated, not forced). IoU code validated (0.1429).
Folded into the benchmark PDF (Table 7). Re-run:
`python scripts/eval_vrsbench_caption_grounding.py --limit 0`.

---


## Session update (2026-09-23)

### P6 — Ablations (section 9.1) COMPLETE ✅ (`scripts/run_ablations.py`)
All four design claims are now measured on real cached data/models (GPU:
RTX 4050). Numbers land in `data/ablation_metrics.json`; the benchmark
report PDF regenerated with a JSON-driven "Ablation study" section
(Tables 4 & 5, pages 3–5).

1. **Optical+SAR fusion vs optical-only on cloud-occluded tiles** —
   SUPPORTED. Ground truth = clean-optical NDWI water; synthetic
   contiguous cloud blinds the optical detector, SAR fills the clouded
   pixels. Mean over 5 SAR sites (assam, sundarbans, dubai, kaziranga,
   pin23260_77410): at 40% cloud fusion IoU 0.70 / recall 0.78 vs
   optical-only 0.44 / 0.44; at 80% cloud fusion recall **64%** vs
   optical-only **17%**. Both 1.000 at 0% cloud by construction.
2. **GSD conditioning on vs off across 0.15–10 m** — SUPPORTED, but
   reframed honestly. `gsd_cond` adds a per-GSD *constant* offset, so at a
   fixed resolution a linear head absorbs it: fresh linear probe on the
   CURRENT encoder is 94.85% (on@10 m) vs 94.89% (off) — ≈equal, by
   design. Its real role is CROSS-resolution: the offset is 45–79% of the
   feature norm and rotates (cosine 0.15 m↔10 m = 0.43, monotone), and it
   flips up to 1.7% of predictions across the range while OFF flips 0%
   (no path from GSD to output). **NOTE:** the `ft_cls_head` EuroSAT
   97.78% is now STALE — the joint encoder passes changed `opt_enc` after
   that head was fit (opt_enc sha differs across
   backbone_ft/pass1/pass2), so the stale head reads ~80% and a fresh
   probe (94.9%) is the honest current EuroSAT linear-probe number.
3. **Router v2 vs v1 on held-out paraphrases** — SUPPORTED. 38 unseen
   paraphrases (asserted absent from `train_router.CORPUS`): v2 (trained)
   **89.5%** vs v1 (rules) **71.1%**, +18.4 pts. Sharpest on the
   change/change-qa boundary the router docstring flags: change-qa v1
   **0%** → v2 **100%** (rules conflate polarity paraphrases).
4. **Mask-derived change-qa vs generative head on CDVQA** — BOUNDED
   (honest). On RGB-only CDVQA `change_or_not` (n=3,493 class-named), a
   per-class RGB-proxy mask sits at the majority floor (53.7% vs floor
   54.2%) and LOSES to the learned scorer (59.2% zero-shot). Why: the
   design's mask-derived change-qa needs per-class *spectral* masks
   (NDWI/NDVI/NDBI) that RGB lacks; the claim's advantage is in the
   multispectral setting (ablation 1 shows the mask/fusion path working on
   real S2/S1), not on RGB CDVQA. Reported truthfully rather than spun.

Harness notes: ablation 2 fits fresh linear probes on current-encoder
features (constant-offset absorption ⇒ on≈off is expected); ablation 4
reuses the checkpoint-sha-stamped `feat_cdvqa_test_mean.<sha>.npz` cache
for the scorer and reads tar-shard RGB for the mask baseline (threshold
calibrated on train shards, applied to test). Re-run:
`python scripts/run_ablations.py` (all four) or `--only 1,3`.

---

## Session update (2026-09-15)

- **Historical change comparison COMPLETE** — year/history questions now ask
  the backend for the earliest verified Sentinel-2 scene, pair it with the
  latest available scene, co-register it, and return a rendered side-by-side
  PNG with spectral-change pixels marked. New API fields:
  `visual_url` and `comparison_url`.
- **Explainable grounding COMPLETE** — the AI console has Box / SHAP / LIME
  evidence modes. SHAP uses the installed package when compatible and a
  Numba-free Shapley superpixel approximation otherwise; LIME explains the
  auditable spectral target detector.
- **Vegetation label bug FIXED** — grounding overlays now use the requested
  target and backend geographic component boxes; a vegetation query no longer
  inherits the old hard-coded water label.
- **"Check for water bodies" routing FIXED** — `check for`, `detect`, and
  `show me` now route to grounding when they name a supported target, while
  change phrases still route to change. A genuine no-water result is answered
  instead of silently falling back to the demo site's synthetic water polygon.
- **Presence-question routing FIXED globally** — `check if`, `is there`,
  `are there`, and `any <target>` now route to VQA for any supported target.
  Verified both a no-water Delhi answer and a yes-water Kaziranga answer;
  temporal increase/decrease questions still route to change QA.

---

## Session wrap-up (2026-09-12)

### Joint training pass 2 COMPLETE + verified ✅ (`scripts/train_joint.py`)
- Pass 2 (warm-start from pass 1, expanded corpus 16,000 train / 467 val
  patches, rgb-stretch 0.2 + band-dropout 0.25 modes, EuroSAT replay at
  0.7 weight every 2nd step) ran 9 epochs 13:59–14:39 on the 6 GB GPU.
- Best epoch 9 (composite 0.5087): **VQA val 76.3%** (binary 76.9%,
  MCQ 75.6%), S2/S1 LULC exact ~14.6/15.0%, EuroSAT replay 97.7%.
  Pass-1 val was 78.5% on the same pinned split — pass 2 traded ~2 pts
  of in-distribution val for RGB-recipe robustness (see below).
- Auto-promoted to `models/backbone_ft.pth` + `models/ben_txt_heads.pth`;
  pass-1 artifacts are at `*_pass1.pth` (NOTE: `*_v1.bak.pth` were
  overwritten by pass-2's promotion — the pre-joint TF-IDF v1 scorer
  weights no longer exist on disk; its Sep-10 zero-shot numbers are
  historical only).
- Live wiring verified end-to-end: backend restarted, /health shows
  scorer `satquery-joint-v2` (weights hash 1df64e57, val 0.7628);
  POST /query on sundarbans answers water-presence correctly with
  honest reconciliation (LULC p=0.84 + spectral 38.2% agree, VQA scorer
  dissents p_yes=0.40 and the answer text says so).
- Scorecard re-run with pass-2 weights: router v1+v2 100% on 13 queries,
  pipeline 100% on answered, abstention 15.4% (unchanged — by design).

### Benchmarks re-run with pass-2 encoder ✅ (`data/bench_metrics.json`)
| Benchmark | Zero-shot (was) | Supervised (was) |
|---|---|---|
| RSVQA-LR | 43.18% (47.13) | **80.39%** (77.51) |
| VRSBench | 27.43% (64.70) | **82.04%** (71.09) |
| CDVQA | 47.91% (55.25) | **78.05%** (75.98) |

- Supervised (deployment-realistic protocol) improved on every benchmark.
- Zero-shot regressed vs the superseded pre-joint v1 generalist: the
  joint model is a BEN-semantics specialist, not a generalist. The
  rgb-stretch input mode did NOT fix VRSBench zero-shot (27.4% ≈ 4-option
  chance 25%) — the regression is question-distribution overfit in the
  jointly-trained FiLM scorer, not input preprocessing. CDVQA zero-shot
  partially recovered (41.3% pass-1 → 47.9% pass-2).
- Feature caches are checkpoint-sha stamped; the re-run rebuilt
  `feat_*.<new-sha>.npz` files automatically (old `c5faf68d` caches
  remain for the old encoder).

### Deliverable PDFs regenerated ✅
- `docs/SatQuery_AI_Benchmark_Results.pdf`: new headline (82.04 best),
  corrected per-type prose, methodology updated (MiniLM/joint v2),
  "Reading the numbers honestly" now states the specialization
  trade-off explicitly (zero-shot candid numbers + why supervised is the
  deployment setting + measured 366-pair adaptation recovery).
- `docs/SatQuery_AI_Completion_Report.pdf`: new "Joint fine-tune 3 —
  encoder-unfrozen multi-task pass (v2)" section, updated benchmark
  numbers, date bumped to 12 September.
- User guide date bumped (content unaffected by weights).
- Gotcha: hand-editing these generators can introduce curly-quote string
  delimiters (U+2018) — py_compile before running.

---

## Session wrap-up (2026-09-09 → 09-10 night)

### Public benchmark evaluations COMPLETE ✅ (`scripts/eval_benchmarks.py`)
Final numbers (zero-shot / supervised), all in `data/bench_metrics.json`:

| Benchmark | Zero-shot | Supervised | Test set |
|---|---|---|---|
| RSVQA-LR | 47.13% | **77.51%** | 8,617 questions / 672 images |
| VRSBench | 64.70% | **71.09%** | 37,409 questions / 9,349 images |
| CDVQA | 55.25% | **75.98%** | 5,832 questions / 366 pairs |

- CDVQA pipeline: 150 train + 80 test shards (~16 GB) from the public HF
  mirror (ljx620/CDVQA); 229/230 fetches ok in 144 min (train-00110 is a
  valid small tar, 300 members).
- CDVQA supervised went 55.51% (chance) → 75.98% via two fixes:
  1. **Change-vector features** — [f0−f1, f0+f1] instead of raw concat;
     the difference is then linearly accessible to the MLP's first layer.
  2. **Pair-aware split** — `groups=meta.image_id` in `adapt_scorer()`;
     random row splits leaked identical pair features (each pair carries
     ~45 templated questions). Val accuracy 75.87% now matches test
     75.98%, proving the leak is gone.
- Zero-shot CDVQA re-run on the FULL 5,832-question test set: 55.25%
  (change/not 45.8%, increase 67.7%, decrease 70.9%).
- Deliverable PDFs generated from the same JSON (no hand-typed numbers):
  `docs/SatQuery_AI_Completion_Report.pdf` (4 pages),
  `docs/SatQuery_AI_Benchmark_Results.pdf` (4 pages, includes charts), plus
  `docs/SatQuery_AI_User_Guide.pdf` (6 pages — website walkthrough matching
  the real UI: navigation, imagery layers, mission sites, workspace, Ask-AI,
  time lapse, insights, shortcuts, troubleshooting).
  Generators: `scripts/gen_completion_report.py`,
  `scripts/gen_benchmark_report.py`, `scripts/gen_user_guide.py` +
  `scripts/pdf_style_kit.py`.

### Globe half-ingested-day fix VERIFIED ✅ (docs/index.html)
- GIBS serves HTTP-200 all-black JPEGs for regions whose granules haven't
  arrived (~20 h into the UTC day, by longitude band) — no error signal.
- `getMaskedLayer()` records 12 per-longitude sector completeness
  (`rec.sectors`, mid-latitudes); `composite()` requires worst ≥ .8
  (raised from .55) or steps back a day (max-worst fallback); resolved
  date flows to `S.baseDate` → header/HUD/fetches.
- Zoom pass: `passDateWithContent()` probes the requested bbox at 96 px
  before the big fetch and steps back to the newest date with content;
  HUD shows "captured <date> (latest available)". Verified 0% black
  during a zoom pass; front-hemisphere brightness mean 179 (was 106).
- Fresh-load header confirmed live in browser:
  "VIIRS SNPP · captured 2026-09-08 (latest available)".

---

## Session fixes (2026-09-09 evening)

### Globe back-hemisphere FIXED ✅ (2026-09-10 early, docs/index.html)
- Symptom: the back of the globe rendered flat/washed-out and completely
  different from the front (measured inner-disc detail 0.8 vs 6.2, std ~15
  vs ~38, blue-tinted).
- Root cause: NOT a three.js/material bug. The GIBS VIIRS_SNPP layer for
  2026-09-08 was still half-ingested (everything west of ~67.5°W — all of the
  Americas + Pacific — was pure black no-data 20 h into the UTC day). The
  black-pixel masking turned that hemisphere transparent, exposing the flat
  2004 BlueMarble underpaint, which the acceptance test never caught because
  it only checked GLOBAL cover > 4% (the eastern hemisphere alone passes).
- Fix (3 parts, all in docs/index.html):
  1. `getMaskedLayer()` now records per-longitude completeness — 12 sectors
     over mid-latitudes (±60°) — on every masked layer (`rec.sectors`).
  2. `composite()` rejects any day whose worst longitude sector is < 55%
     covered (floor = the backfill layer when the base has one, since it
     fills the base's gaps from underneath) and steps back a day, up to 3
     tries; if all are incomplete it uses the fullest one.
  3. The resolved date is stored as `S.baseDate` and a new `baseDate(p)`
     helper feeds it to everything that displays or fetches the base layer:
     zoom-pass fetch (`fetchPass`), pass HUD, header, workspace modal init,
     and session export — so zoom-in never regresses to the partial day.
- Verified: fresh load resolves to 2026-09-07 (complete in all 12 sectors),
  header shows "captured 2026-09-07 (latest available)", North America back
  hemisphere now renders detail 6.0 (was 0.8), std 36-46 (was ~15), warm land
  tones matching the front hemisphere.

### Fuse pipeline FIXED — alignment repair ✅
- Root cause: cross-modal pairs measured 4.18 px residual shift; the pipeline
  already repaired it (`align_pair`) but passed the STALE pre-repair
  `pair_check` to the gate, which degraded fuse -> describe.
- Fix in `pipeline.py`: re-run `check_pair` after repair, log an
  `align_repair` trace step (before/after residual: 4.18 px -> 0.14 px).
- Fuse query now executes as fuse at 77.5% confidence, gate 5/5.

### Cloud-shadow QA mask FIXED ✅
- Old heuristic (`nir<0.06 & brightness<0.10`) flagged dark water and
  mangrove swamp as "shadow": Sundarbans 71% masked, water 1.6% detected.
- New heuristic in `ingest.py`: shadow requires NIR-dark AND blue/brighter
  than local neighbourhood (scattering keeps real cloud shadow blue).
- Sundarbans now detects 38.2% water (NDWI ground truth 38.2%), answers at
  66% confidence. Tile cache under `data/tile_cache` was cleared and rebuilt.

### Router v2 retrained again ✅
- "What land cover is dominant here?" misrouted to change; added present-tense
  "land cover" vqa examples. Corpus 336 queries; held-out test 83.8%;
  all 13 eval queries route correctly.

### New site: kaziranga ✅
- Assam patch (26.14, 91.73) is dry upland — zero NDWI water on both dates;
  the water queries abstain there BY DESIGN (honest abstention precision).
- Added Kaziranga NP floodplain (26.58, 93.17): 2 optical + 1 SAR fetched,
  Nov 2024 has 3.2% water in 8 regions, ground query answers at 55% conf.

### Scorecard after fixes ✅ (`data/scorecard.json`)
- Router v1+v2 100% on 13 queries; pipeline 100% on answered; abstention 15.4%
  (2 water queries on assam abstain honestly — no water present; use
  `--site kaziranga` or sundarbans to see water grounding answer).

---

## Training completed (2026-09-09)

### Router v2 retrained — `scripts/train_router.py` ✅
- Corpus grown 59 → 330 queries (6 intents, balanced, eval queries excluded)
- 5-fold CV accuracy **84.9%**, held-out test **78.8%** (was 71% CV on 59)
- All 13 run_eval queries route correctly; old router backed up to `models/router_v2.seed59.bak.pkl`
- `models/router_v2.pkl` — new artifact with metrics dict

### Backbone fine-tuned on EuroSAT — `scripts/train_backbone.py` ✅
- EuroSAT (Sentinel-2 13-band, 10 classes): 16,200 train / 5,400 val / 5,400 test
- Init: torchgeo SENTINEL2_ALL_MOCO; trained opt_enc + band_proj (lr 1e-5), gsd_cond + cls head (lr 1e-3)
- 6 epochs on RTX 4050, ~10 min; **held-out test accuracy 97.78%** (per-class recall in `data/backbone_train_metrics.json`)
- Checkpoint `models/backbone_ft.pth` (202 MB); `get_backbone()` now auto-loads it (backbone id `resnet50-s2-moco-finetuned-eurosat`)
- EuroSAT data cached at `data/eurosat/` (~2 GB)

### Scorecard after training — `data/scorecard.json`
- Router v2 92.3% on 13 queries, pipeline 80% on answered, abstention 23.1%
- Known issues: "What land cover is dominant here?" misroutes to change; fuse query executes as describe in pipeline (investigating)

### NOTE on dataset choice — RESOLVED
EuroSAT used for the initial fine-tune (PS clause allows "BigEarthNet.txt **or any open source training data**").
The BigEarthNet.txt imagery question was then solved WITHOUT the 170+ GB LMDB:
the annotations parquet (0.47 GB, HF `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`) pairs
with the classic torchgeo multipart tars (`torchgeo/bigearthnet` — S2 63.5 GB,
S1 54.8 GB), from which we **selectively streamed only 7,084 patches**
(round-robin by country/snow) instead of extracting all 464,044. Both tar
archives deleted after extraction; ~260 GB free after the full S2+S1 chain.

### BigEarthNet.txt fine-tune COMPLETE (2026-09-10 night) ✅
Phase 1 — S2 optical (`fetch_ben_txt.py`, `train_ben_txt.py`, `eval_ben_txt.py`):
- 6,861 S2 patches on disk (221 of 7,084 absent from tar — Finland/Portugal
  quality-filtered at source), 142,478 annotation rows subset
- Encoder fine-tuned on BEN-19 multi-label BCE (loss 0.31 -> 0.23), then
  feature-cached head training: 87,452 VQA examples, **val 62.2%**;
  caption head Linear(2048->19) on 5,843 patches (val exact 14.0% — strict
  full-19-set match, train still improving at the 6GB epoch budget)
- Official bench split: **55.5%** over 11,703 binary/mcq (weighted chance ~39%);
  by category: climate zone 81.3%, country 70.1%, relative pos 31.9%
  (`data/ben_bench_metrics.json`)
Phase 2 — S1 SAR (`fetch_ben_s1.py`, `train_ben_s1.py`):
- Same 6,861 patches' VV/VH extracted via the torchgeo `s1_name` mapping
  (dB, mean-subtracted VV -12.64 / VH -19.35), 0 errors
- `sar_enc` fine-tuned on BEN-19 (optical frozen): val loss 0.254,
  exact-match 14.0% and still rising at epoch 4 — tiny 6GB-GPU budget, not saturated
- Saved back into `models/backbone_ft.pth` (opt_enc + sar_enc + heads, 657 keys);
  smoke-tested: optical path (1,2048) and real-patch SAR path (1,2048) both run
- Bench note: the bench eval is S2-only (bench split patches' S1 imagery was
  not downloaded); SAR quality is attested by val loss + a trained s1_cls_head
Pipelines/scripts (idempotent, resumable): `download_ben_s2/s1.py`
(Range-header resumable), `run_ben_pipeline.py` orchestrator.
Windows-specific fixes this session: `_ConcatParts` must subclass
`io.RawIOBase` (tarfile calls `.read()`); close the stream before deleting
tar parts (WinError 32 file lock).

---

## What is built and working right now

### Stage 0 — Ingest, sensor abstraction, normalisation ✅
- `satquery/contracts.py` — all typed contracts (Tile, ExecPlan, TaskResult, TraceRecord, etc.)
- `satquery/stage0/profiles.py` — sensor profile registry: S2 MSI, S1 GRD, Cartosat-2S PAN/MX, RISAT SAR, generic RGB, RSVQA-LR
- `satquery/stage0/ingest.py` — ordered normaliser sequence (open/validate, profile mapping, SAR dB+speckle, optical stretch, QA mask, tile, cache), pair checks (bi-temporal + cross-modal phase-correlation alignment), TileCache

### Real imagery corpus ✅
- `scripts/fetch_data.py` — fetches Sentinel-2 L2A (Earth Search) + Sentinel-1 RTC (Planetary Computer)
- `data/raw/manifest.json` — 9 sites × 2 dates S2 + S1 SAR
  - Sites: assam, delhi, sundarbans, punjab, gangotri, rondonia, creekfire, nile, dubai
  - All at 0% cloud (dry season passes) except assam t2 (1.3%)

### Stage 1 — Agentic controller ✅
- `satquery/stage1/router.py` — v1 rule router + v2 TF-IDF LogReg classifier
- `satquery/stage1/gate.py` — 5-field compatibility gate (number/modality/format/metadata/compatibility)
- `satquery/stage1/registry.py` — declarative ToolSpec registry, gsd_fit selector, param binding enforcer
- `satquery/stage1/controller.py` — sequences all six controller duties, each logged to trace
- `models/router_v2.pkl` — trained classifier; 100% on the 5 PS representative queries

### Stage 2 — Specialist heads ✅
- `satquery/stage2/backbone.py` — ResNet50 SSL-pretrained on Sentinel-2 (torchgeo SENTINEL2_ALL_MOCO); 12-ch band projector, GSD conditioning MLP, SAR encoder (SENTINEL1_ALL_MOCO)
- `satquery/stage2/heads.py` — VQAHead, CaptionHead, GroundHead, ChangeHead (change + change-qa derived), FuseHead; all return TaskResult with text/visual/confidence/raw
- All six intents produce real output on real Assam pixels

### Stage 3 — Integration, confidence, abstention ✅
- `satquery/stage3/integrate.py` — temperature-scaling calibration per task, cross-model conflict detection, abstention thresholds, merged Answer

### Stage 4 — Trace + evidence + PDF ✅
- `satquery/stage4/report.py` — PDF report via reportlab rendered from TraceRecord; includes gate verdicts, calibration, reproducibility section (input_hash + weights_hash)
- PDF tested and generates at `docs/Prototype/demo_report.pdf`

### Stage 5 — FastAPI backend ✅
- `satquery/api.py` — POST /upload, POST /query, GET /report/{run_id}, GET /health, GET /sites
- Running on `http://localhost:7860`
- All 9 manifest sites auto-resolve tile paths from site_id
- Verified: `/query` with site_id=assam returns real inference in < 1s (cached)

### Full pipeline runner ✅
- `satquery/pipeline.py` — `run_query()` wires Stage 0→1→2→3→4 end-to-end
- Tested on all 6 intents on Assam pixels

### Frontend wiring ✅
- `docs/index.html` — exact same design unchanged; new `<script>` block at the end
  - On page load, probes `http://localhost:7860/health`
  - If backend alive: replaces `ask()` with `askReal()` (live POST /query)
  - `askReal()` resolves tiles from manifest, streams live agent step animations, shows real answer + PDF report link
  - Falls back to demo stub on error
  - Conflicts shown, confidence donut updated with real value

### Evaluation harness + router training ✅
- `scripts/fetch_data.py` — data acquisition
- `scripts/run_eval.py` — router accuracy, pipeline accuracy, abstention rate, confusion matrix, JSON scorecard
- `data/scorecard.json` — current run: router v1+v2 both 100% on 5 PS queries; pipeline 75% answered; abstention 20%
- `START_DEMO.bat` — one-click launcher (starts backend + opens browser)

---

## What still needs to be done (continuation points)

### P6 — Ablations (section 9.1) — DONE ✅ (2026-09-23)
All four claims measured; see the 2026-09-23 session update above and
`data/ablation_metrics.json` (harness: `scripts/run_ablations.py`; results
folded into `docs/SatQuery_AI_Benchmark_Results.pdf`, Tables 4–5):
1. Optical+SAR fusion vs optical-only on cloud-occluded tiles — SUPPORTED
   (at 80% cloud, fusion recall 64% vs optical-only 17%).
2. GSD conditioning on vs off across 0.15–10 m — SUPPORTED (cross-resolution
   behaviour; fixed-GSD is a constant offset a linear head absorbs).
3. Router v2 vs v1 on held-out paraphrases — SUPPORTED (89.5% vs 71.1%).
4. Mask-derived change-qa vs generative head on CDVQA — BOUNDED (learned head
   wins on RGB CDVQA; the mask method's edge is the multispectral setting).

### Benchmark coverage — partially done
- DONE: RSVQA-LR, VRSBench, CDVQA (zero-shot + supervised; see
  `docs/SatQuery_AI_Benchmark_Results.pdf`)
- DONE (2026-09-24): VRSBench caption metrics (BLEU/ROUGE/CIDEr) + grounding
  coverage — both OOD, honest bound (Table 7; see the session update above)
- DONE (2026-09-25): RSVQA-HR test set 1, LEVIR-CD (binary) change-mask F1+IoU,
  and BigEarthNet held-out fuse per-class IoU — all folded into the benchmark
  PDF. Still open: RSVQA-HR test_phili split and the LEVIR-CD / SECOND
  multi-class set (each needs more compute or a download)

### SAR data gap
- gangotri, rondonia, creekfire, nile — no SAR tile fetched (Planetary Computer had no coverage)
- Fuse intent auto-degrades to optical-only for these sites

### Calibration fitting — DONE ✅ (2026-09-24)
- `scripts/fit_calibration.py` fits per-task temperature scaling and reports
  ECE on held-out benchmark splits → `data/calibration_metrics.json`.
- `integrate.DEFAULT_TEMPERATURES` now fitted for vqa (2.11) + change-qa
  (1.18); other tasks kept as documented priors. See the 2026-09-24 session
  update above. Zero-shot scorer confidence found uninformative (honest
  result); adapted scorer calibrates cleanly.

### Offline packaging
- CDN dependencies (three.js, fonts) still require internet
- Planned: vendor JS + pre-warm tile cache + ONNX quantised models

---

## How to resume

```bash
cd C:\Users\Sharanya\Documents\Hackathon\SatQuery_AI_SIH26167

# Start backend
python -m uvicorn satquery.api:app --port 7860 --reload

# Open demo
start docs\index.html

# Or double-click:
START_DEMO.bat

# Re-run eval
python scripts/run_eval.py --quick

# Fetch more data
python scripts/fetch_data.py --force

# Train router
python scripts/train_router.py  # router v2 -> models/router_v2.pkl
```

## Key files

| Path | What it is |
|------|------------|
| `satquery/contracts.py` | All typed contracts |
| `satquery/pipeline.py` | End-to-end run_query() |
| `satquery/api.py` | FastAPI backend |
| `satquery/stage0/` | Ingest + profiles + pair checks |
| `satquery/stage1/` | Controller (router+gate+registry) |
| `satquery/stage2/` | Backbone + 6 specialist heads |
| `satquery/stage3/integrate.py` | Calibration + conflicts + abstention |
| `satquery/stage4/report.py` | PDF report renderer |
| `docs/index.html` | Frontend (exact original design + real wiring) |
| `data/raw/manifest.json` | Real satellite data for all 9 sites |
| `models/router_v2.pkl` | Trained 6-way router |
| `data/scorecard.json` | Latest evaluation results |
| `docs/Prototype/demo_report.pdf` | Sample generated PDF report |
