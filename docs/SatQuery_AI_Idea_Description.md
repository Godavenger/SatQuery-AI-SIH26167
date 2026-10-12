# SatQuery AI - Ask the Satellite: Explainable Multimodal Vision-Language Earth-Observation Assistant

## 1. One-Line Pitch

SatQuery AI lets anyone ask a satellite a question in plain language - 'Did the flood extent grow since March?', 'Is there water near this village?', 'How much farmland became built-up area last year?' - and returns an evidence-grounded answer in seconds: a calibrated confidence score, geo-anchored visual overlays, the underlying spectral measurements, and a downloadable audit report. It turns satellite imagery from a specialist GIS workflow into a conversation.

## 2. The Problem We Solve

Earth observation is drowning in data and starving for answers. Sentinel-2 alone re-images the entire globe every five days, and ISRO's Cartosat, RISAT, Resourcesat and NISAR-era constellations add ever-growing volumes. Yet extracting one simple fact - has this lake shrunk, is this field flooded, did built-up area expand - still demands a trained analyst, the right software, knowledge of spectral indices and thresholds, and hours of manual work. Three barriers keep satellite intelligence away from the people who need it most:

- **The expertise barrier.** District officers, disaster responders, municipal bodies, panchayats and NGO field teams cannot operate desktop GIS, choose indices, tune thresholds or pick models. They have questions, not workflows.
- **The modality barrier.** Optical imagery is blinded by cloud exactly when it matters most - floods, cyclones, monsoon agriculture. SAR penetrates cloud and works day and night, but almost no general-purpose AI assistant understands optical and SAR together.
- **The trust barrier.** Generic vision-language models produce fluent prose with no evidence, no calibrated confidence and no audit trail. An operational report cannot cite 'the model said so'; decisions need to see where an answer came from and how certain it is.

SIH26167 asks for an interactive vision-language assistant that performs multimodal remote-sensing image analysis through text queries. SatQuery AI is our answer - an agentic, evidence-first system that answers, shows and proves.

## 3. The Proposed Solution

SatQuery AI is an agentic vision-language assistant for satellite imagery. A user types a natural-language question over any location on an interactive 3-D globe. An agentic controller - deliberately not a generic LLM - interprets the query, validates it against a five-field compatibility gate, selects the right specialist model from a declarative registry, executes it on canonicalised, sensor-agnostic image tiles, and merges the outputs into a calibrated, evidence-backed answer.

Six capabilities live behind one interface:

1. **Visual Question Answering (VQA)** - 'Is there a water body in this image?' returns a yes/no answer with calibrated confidence and the land-cover evidence behind it.
2. **Scene Description** - 'Describe this scene.' returns a grounded natural-language summary with land-cover composition.
3. **Grounding / Detection** - 'Show me the vegetation.' returns geo-anchored boxes and masks rendered on the map, with class areas.
4. **Change Detection** - 'What changed between these two dates?' returns a change mask, area statistics and a rendered side-by-side comparison with spectral-change pixels highlighted.
5. **Change QA** - 'Did water increase or decrease?' returns a categorical answer derived from measured change areas, not guesswork.
6. **Optical-SAR Fusion** - 'Fuse the optical and radar views.' returns a combined product that sees through cloud and at night.

Every answer can carry four pieces of evidence: a visual overlay, a calibrated confidence value, a machine-readable trace of every processing step, and a one-click PDF report. Where the signals behind an answer disagree, the disagreement is surfaced; where a precondition fails, the system declines and names the failed check rather than guessing.

## 4. How It Works - The Six-Stage Pipeline

**Stage 0 - Ingestion and sensor abstraction.** GeoTIFFs from any supported sensor enter a canonical Tile contract: semantic band slots (blue, green, red, NIR, SWIR, VV/VH and more), ground-sample distance, CRS, affine transform, acquisition timestamp and a quality mask. Optical bands are normalised; SAR tiles are despeckled with a refined-Lee filter; Sentinel-2 SCL layers drive cloud and cloud-shadow masks. Bi-temporal and cross-modal pairs are checked and, when needed, co-registered by phase correlation - on a real Sentinel-2 pair this repair reduced residual misalignment from 4.18 px to 0.14 px. Cross-modal optical+SAR pairs are deliberately not 'repaired' against each other, because phase correlation between modalities measures texture disagreement rather than misregistration.

**Stage 1 - Agentic controller.** A trained six-way intent router (second generation, about 85% five-fold cross-validation and 100% on the live evaluation suite) classifies the query into one of the six intents, and slot extraction pulls out targets, dates, directions and comparison phrases. A five-field compatibility gate then checks the number of images, modality, format, metadata and task compatibility before anything runs. A declarative tool registry filters specialists by task, modality and GSD range, and parameter binding is restricted to each tool's permitted parameters. A failed gate short-circuits to a reasoned abstention - which is itself a complete, scoreable trace.

**Stage 2 - Specialist registry.** Six heads share one remotely-sensed-adapted backbone - a ResNet-50 self-supervised-pretrained on Sentinel-2 imagery, with a 12-channel band projector, a GSD-conditioning MLP and a SAR encoder - so one adapted model serves many tasks and resolution regimes. Task heads are swappable, small and versioned behind a single run() contract, which lets a new specialist join the system without touching the controller.

**Stage 3 - Integration, confidence and abstention.** Per-task temperature scaling calibrates confidence so that '0.8' means the same thing across every head. Each answer reconciles the independent signals behind it - an auditable spectral measurement, a trained land-cover classifier and the trained answer scorer (and, in fusion, the optical and SAR heads) - and when they disagree the merged answer states the conflict honestly and drops its confidence instead of hiding it; the integration layer also carries a cross-specialist conflict check (grounding vs VQA, change vs fusion) that activates on any query routed through more than one head. Abstention thresholds then decide whether to answer or decline.

**Stage 4 - Trace, evidence and reporting.** Every step emits a TraceStep with sequence number, tool name and version, weights hash, parameters, input hash, output summary, confidence, latency and gate verdict. The same record renders the live view, the JSON API response and a downloadable PDF report - so any reviewer can re-run and verify an answer.

**Stage 5 - Serving and interface.** A FastAPI backend exposes /upload, /query, /report and /health, and drives a browser front-end: an interactive 3-D globe with daily NASA satellite mosaics, eight switchable base layers (VIIRS and MODIS true colour and false colour, Black Marble night lights), scientific overlays (aerosol optical depth, cloud-top temperature, precipitable water, sea-surface temperature, chlorophyll-a, snow cover, sea ice, active fires), automatic zoom from globe view through regional passes to sub-metre aerial imagery, six feature benches, and a live agent-trace panel that animates each processing step as it happens.

**Stage 6 - Evaluation harness.** The system is evaluated end-to-end on public benchmarks from day one - accuracy by question type, router accuracy, pipeline accuracy, abstention rate and calibration - with results written to machine-readable scorecards.

## 5. Multimodal by Design

Sensor heterogeneity is handled at the architecture level, not patched per dataset. The semantic band-slot abstraction means downstream stages never branch on platform: a Cartosat panchromatic scene, a Sentinel-2 MSI stack and a RISAT SAR product all arrive as the same canonical Tile. Ground-sample distance is a first-class conditioning input across 0.3-10 m, so the model knows whether it is looking at 10-m pixels or sub-metre ones. The joint training corpus mixes Sentinel-1 SAR and Sentinel-2 optical with text (BigEarthNet.txt), and input-mode augmentation - band dropout and RGB-stretch - teaches the encoder to handle partial-band public imagery and contrast-stretched RGB inputs. When SAR coverage is unavailable for a site, fusion queries degrade gracefully to optical-only and say so in the answer.

## 6. Explainability, Calibration and Honest Abstention - The Trust Layer

This is what separates SatQuery AI from a chatbot with a satellite image attached:

- **Grounded evidence.** Answers come with geo-anchored boxes, masks, change overlays, class areas and rendered before/after comparisons - the answer is drawn on the map, not merely asserted.
- **Interpretable AI console.** The console offers three evidence modes - Box, SHAP and LIME - so a reviewer can see which regions and spectral features drove a decision. SHAP runs with the installed package where compatible and a Numba-free Shapley superpixel approximation otherwise; LIME explains the auditable spectral target detector.
- **Calibrated confidence.** Temperature scaling per task makes confidence comparable across heads, and the confidence is displayed with every answer.
- **Cross-signal reconciliation.** Each answer weighs the independent signals behind it - an auditable spectral measurement, a trained land-cover classifier and the trained answer scorer - and when they disagree the answer says so and drops confidence (on the Sundarbans, for instance, the land-cover head and the 38.2% spectral water fraction agree while the VQA scorer dissents, and the answer states it). Fusion reconciles the optical and SAR heads the same way, and the integration layer carries a cross-specialist conflict check for queries that run more than one head. A single VLM cannot offer this, because it has no independent signals to cross-examine.
- **Honest abstention.** An unmet precondition - one image where two are required, a cloud-covered tile, an unsupported task - triggers a reasoned decline naming the failed check. The system treats 'I cannot answer that reliably' as a first-class, correct answer.
- **Full auditability.** Input hashes, weights hashes, tool versions, parameters and gate verdicts are logged in every trace; the same record powers the live view, the API response and the PDF report.

## 7. The Working Prototype

The prototype is a complete, running system, not a mock-up:

- A browser GUI with an interactive 3-D globe - orbit, zoom and pan; automatic hand-off from whole-Earth daily mosaics through regional passes to sub-metre aerial imagery; search by place name, mission site or raw coordinates; and shift-click anywhere on Earth to jump straight to the analysis workspace at that point.
- Ten real satellite sites on genuine Copernicus Sentinel-2 optical imagery: Assam flood plains, Delhi, the Sundarbans, Punjab, Gangotri, Rondonia (deforestation), a wildfire creek site, the Nile, Dubai and Kaziranga - six of them also carrying Sentinel-1 SAR, with bi-temporal pairs wherever change analysis applies; the four sites without a SAR tile demonstrate the pipeline's graceful optical-only degradation.
- A live AI console: type a question, watch the agent trace animate through interpret, check, select, execute, combine and evidence steps, and receive the answer with overlays, confidence and a downloadable PDF report.
- Historical change comparison: year and history questions automatically fetch the earliest verified Sentinel-2 scene, pair it with the latest, co-register the two and render a side-by-side image with spectral-change pixels marked.
- Explainable grounding with Box / SHAP / LIME evidence modes, and honest presence-question routing - a genuine 'no water here' is answered as 'no water', with each specialist's agreement or dissent stated.

## 8. Measured Results

All numbers below come from the project's own evaluation artifacts:

- **Router (six-intent query classifier):** 100% accuracy on the live evaluation suite; about 85% five-fold cross-validation on held-out paraphrases.
- **VQA validation:** 76.3% on the held-out question set (binary 76.9%, multiple-choice 75.6%) after joint multimodal fine-tuning.
- **Public benchmark scorers (supervised, deployment-realistic protocol):** RSVQA-LR 80.8% (8,617 questions), VRSBench 83.8% (37,409 questions), CDVQA 77.2% (5,832 image pairs).
- **Domain replay:** 97.7% on EuroSAT classification replay during joint training, showing the adaptation kept core land-cover competence.
- **Alignment repair:** residual pair misalignment reduced from 4.18 px to 0.14 px by the co-registration step on a real Sentinel-2 pair.
- **Pipeline scorecard:** 100% router accuracy and 100% pipeline accuracy on answered queries across the six intents; abstention is a deliberate, measured behaviour (15-20% on the hardest suite) and its precision is a reported metric.

## 9. Technology Stack

- **Vision-language (current build):** a ResNet-50 self-supervised Sentinel-2 backbone (torchgeo MoCo) with a 12-channel band projector, a GSD-conditioning MLP and a Sentinel-1 SAR encoder; MiniLM-L6 sentence embeddings for the question; a FiLM-conditioned option-scoring VQA head and a BigEarthNet-19 land-cover caption head. (RemoteCLIP / SigLIP / LLaVA-class encoders and QLoRA adaptation are reference architectures on the roadmap, not the shipped build.)
- **Specialist methods (current build):** grounding and detection by spectral-index classification (NDWI / NDVI / NDBI / NBR) with connected-component regions; change detection by change-vector analysis over per-class spectral deltas; optical-SAR fusion by late fusion of the two land-cover heads. Discrete-object detection (ships, aircraft, vehicles) is deliberately out of scope - the gate abstains on it rather than guessing. (ChangeFormer / BiT, CLIPSeg + SAM and open-vocabulary detectors are reference heads for the roadmap.)
- **Geospatial:** rasterio / GDAL, pyproj, refined-Lee speckle filtering, phase-correlation co-registration, Sentinel-2 SCL cloud/shadow masking, spectral index measurement.
- **Backend:** Python 3.12, FastAPI, PyTorch, ONNX Runtime and int8 quantisation targets, LRU model registry, tile caching.
- **Frontend:** HTML/CSS/JS, three.js 3-D globe, Leaflet with Esri World Imagery, canvas overlays, vendored assets for offline use.
- **Data and evaluation:** BigEarthNet.txt, VRSBench, RSVQA LR/HR, CDVQA, LEVIR-CD / OSCD / SECOND, DOTA / DIOR / LoveDa, ISRO / SAC evaluation pairs.

## 10. Use Cases and Impact

- **Agriculture and water:** crop-area change, irrigation and water-body extent answered in a sentence, with the mask and the areas behind the claim - no GIS analyst needed.
- **Disaster response:** SAR sees through cloud and at night, so flood and damage extent are available when optical satellites are useless; before/after comparisons render automatically.
- **Urban and infrastructure:** 'Has built-up area increased?' returns the change mask, the hectares and the confidence - supporting planning and encroachment monitoring.
- **Forest and environment:** deforestation, land-cover shift and wetland shrinkage tracked from open multitemporal imagery at query speed.
- **Education and research:** students and researchers get six tasks behind one contract - a new specialist can be added behind the same run() signature without touching the controller.

## 11. Target Users

- **Non-expert decision-makers** - district officers, panchayat and municipal staff, NGO field teams: satellite evidence becomes a question in plain language instead of a workflow, a model choice and a parameter set.
- **ISRO / SAC and state remote-sensing centres** - the auditable trace (task, model, version, weights hash, parameters, confidence) makes an AI answer defensible in an operational report, which is precisely what blocks adoption today.
- **Analysts** - routine questions are answered and evidenced automatically; expert time goes to the cases the system declines, which it names explicitly.
- **Researchers and students** - one registry, one contract, six tasks: an extensible research platform, not a black box.

## 12. Innovation and Uniqueness

1. **Agentic controller, not a monolithic VLM.** Interpretation, gating, specialist selection, execution and integration are separate, auditable steps - each one emits its own trace line.
2. **Cross-signal reconciliation.** Every answer is cross-examined by the independent signals behind it - a measured spectral fraction, a trained land-cover classifier and the trained answer scorer (optical vs SAR in fusion) - and their disagreement drops confidence and is surfaced in the answer. A single VLM cannot offer this.
3. **Resolution-gap engineering.** Trained largely at 10 m Sentinel resolution and conditioned on GSD across 0.3-10 m, the system is designed to serve ISRO's sub-metre Cartosat and RISAT imagery.
4. **Abstention as a capability.** A reasoned decline naming the failed check is treated as a correct answer, and abstention precision is a reported metric.
5. **Sensor-agnostic canonical tiles.** Band slots, GSD, CRS and QA masks make every downstream stage platform-independent; adding a sensor is a profile, not a rewrite.
6. **Evidence-first UX.** Overlays, SHAP / LIME explanations, calibrated confidence, hash-logged traces and PDF reports are the product, not an afterthought.

## 13. Future Roadmap

The roadmap deepens SatQuery AI along eight axes:

1. **Native ISRO constellation integration.** Sensor profiles and ingestion adapters for Cartosat-2S / Cartosat-3, RISAT-1A / EOS-04, Resourcesat-2 / 2A (AWiFS, LISS-III, LISS-IV) and NISAR, wired directly to Bhoonidhi and MOSDAC portals. The profile registry already makes each new sensor a configuration rather than a code change.
2. **Voice and Indian-language queries.** Speech input and output, and query understanding in Hindi and major regional languages, so a field officer can ask a question by voice in their own language and receive the same evidence-backed answer.
3. **Continuous monitoring and alerting.** Area subscriptions that watch new acquisitions automatically, run change detection on each fresh scene and push evidence-backed alerts - flood spread, deforestation, encroachment, reservoir shrinkage, fire scars - with the overlay, confidence and report attached.
4. **Edge, offline and mobile deployment.** ONNX int8-quantised heads, vendored assets, a pre-warmed tile cache and a fully air-gapped package for district hardware, plus a progressive web app for field use where connectivity is intermittent.
5. **Time-series and deeper analytics.** Multi-date stacks, per-class area timelines, trend charts, zonal statistics and anomaly detection, so questions like 'how has this lake changed over five years?' return a chart as well as a sentence.
6. **Expanded evaluation and calibration programme.** RSVQA-HR, captioning metrics (CIDEr / BLEU-4 / ROUGE-L), grounding IoU@0.5 on VRSBench, LEVIR-CD / SECOND change-mask F1 and IoU, formal ECE calibration on held-out splits, and ablations quantifying the optical+SAR fusion benefit under cloud, GSD conditioning, and router paraphrase robustness.
7. **Ecosystem APIs and sector packs.** A clean REST API for state dashboards, disaster-response consoles and GIS platforms; sector packs for agriculture, forestry, water resources, urban planning and disaster management that bundle indices, thresholds and report templates behind the same query interface.
8. **Model governance and continuous improvement.** A versioned model registry with weights-hash provenance (already logged per trace), drift monitoring, continuous evaluation on fresh acquisitions, and on-prem deployment options for state remote-sensing centres that require data sovereignty.

## 14. Feasibility and Sustainability

The system is built on open data (Copernicus Sentinel-1/2, BigEarthNet.txt, VRSBench, RSVQA, CDVQA, LEVIR-CD, DOTA, DIOR, LoveDa) and open tooling (PyTorch, FastAPI, GDAL, ONNX Runtime), so there are no licence blockers and no procurement dependency. The router and calibrated heads run on CPU; the QLoRA adaptation path keeps training within a single consumer GPU's reach; and the six-stage architecture lets every component be developed, tested and swapped independently. The same registry contract that carries six specialists today carries the roadmap's new sensors, languages and sector packs - growth is additive, not a rewrite. Operationally, SatQuery AI shifts analyst hours from routine extraction to judgement, and its audit trail is designed for the reporting standards of exactly the institutions that will deploy it.

## 15. Conclusion

SatQuery AI turns satellite archives into a conversation you can trust. It pairs the accessibility of natural-language queries with the rigour of a gated, traced, calibrated pipeline; it fuses optical and radar so cloud cover stops being a deadline; and it shows its work - overlays, measurements, confidence and a hash-logged trail - so its answers can live inside an official report. Ask the satellite. It answers, it shows, and it proves.
