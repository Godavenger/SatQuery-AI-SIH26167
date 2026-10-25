# Runtime asset provenance

The v1.1.0 GitHub Release includes `satquery-runtime-v1.1.0.zip`. Source code remains
in Git; large inference files are published as a release asset. `runtime-assets.json`
records the archive checksum and each file's size and SHA-256. Install with
`python scripts/runtime_assets.py install` and verify with the `verify` subcommand.

## Contents

| Files | Origin and purpose |
|---|---|
| `models/backbone_ft.pth` | Locally adapted Sentinel optical/SAR encoder used by the project |
| `models/ben_txt_heads.pth` | Joint v2 VQA, caption and SAR classification heads |
| `models/router_v2.pkl` | Locally trained TF-IDF/logistic-regression intent router |
| `models/minilm/` | `sentence-transformers/all-MiniLM-L6-v2`, snapshot `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; includes model card and Apache-2.0 license |
| `models/onnx/*.int8.onnx` | Four project-exported quantized optical/SAR feature and land-cover graphs |
| `data/raw/` | Prepared Sentinel-2 and Sentinel-1 imagery, masks, and portable scene manifest for ten mission sites |

The release excludes personal pinned-location history, uploads, result traces,
training intermediates, redundant checkpoint backups and large benchmark corpora.
Caches are rebuilt locally. `scripts/fetch_*.py` and training/evaluation scripts
remain available for reproducing the research workflow.

## Upstream attribution

- MiniLM: [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
  Apache License 2.0. Its original model card and full license accompany the weights.
- Optical imagery contains modified Copernicus Sentinel data (2020–2025), acquired
  through [Earth Search](https://earth-search.aws.element84.com/v1). Per-scene STAC
  identifiers, dates and footprints are retained in the manifest.
- Sentinel-1 RTC imagery was acquired through the
  [Microsoft Planetary Computer Sentinel-1 RTC collection](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc).
  Its upstream processing and data terms continue to apply.
- The adaptation workflow uses
  [TorchGeo](https://github.com/microsoft/torchgeo),
  [BigEarthNet](https://bigearth.net/) and
  [BigEarthNet.txt](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt).
  Consult those projects for pretrained-model and training-data provenance; the
  project's MIT source-code license does not replace upstream terms.

The installer accepts assets only from this repository's GitHub Releases URL and
verifies hashes before replacing files. Different existing artifacts are preserved
unless `--overwrite` is explicitly supplied. Load pickle/PyTorch artifacts only
from trusted sources; the pinned release is the supported distribution.

## Maintainer rebuild

With the trained artifacts, mission imagery and a complete local MiniLM directory:

```bash
python scripts/runtime_assets.py build --version v1.1.0 --minilm-dir models/minilm
```

This writes the archive and `SHA256SUMS` under ignored `dist/`, and refreshes the
tracked inventory. Validate installation into an empty destination before creating
a new release. Every version must identify the exact source commit and its assets.
