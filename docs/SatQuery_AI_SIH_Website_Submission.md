# SatQuery AI - SIH Website Submission (Copy-Paste Ready)

Problem Statement ID: SIH26167
Theme: Space Technology
PS Category: Software

Title (under 100 characters):
SatQuery AI - Ask the Satellite: Explainable Multimodal Vision-Language Earth-Observation Assistant

The sections below match the SIH idea-submission fields. Copy each block into the corresponding field on the portal. All text is plain-text friendly (no markdown symbols that break web forms). A condensed single-box version is included at the end in case the portal asks for one combined description.

=====================================================================
FIELD 1: PROPOSED SOLUTION
=====================================================================

SatQuery AI is an agentic vision-language assistant that answers plain-language questions over satellite imagery. A user types a question - 'Did the flood extent grow since March?', 'Is there water near this village?', 'How much farmland became built-up last year?' - and the system returns an evidence-grounded answer in seconds: a calibrated confidence score, geo-anchored visual overlays, the underlying spectral measurements, and a downloadable audit report.

Instead of one monolithic model, an agentic controller interprets the query, validates it against a five-field compatibility gate, selects the right specialist model from a declarative registry, executes it on canonicalised sensor-agnostic image tiles, and merges the outputs with multi-signal reconciliation and honest abstention. Six capabilities live behind one interface:

1. Visual Question Answering - yes/no and categorical answers with calibrated confidence and land-cover evidence.
2. Scene Description - grounded natural-language summaries with land-cover composition.
3. Grounding / Detection - geo-anchored boxes and masks rendered on the map, with class areas.
4. Change Detection - change mask, area statistics and a rendered side-by-side before/after comparison with spectral-change pixels highlighted.
5. Change QA - increase/decrease answers derived from measured change areas, not guesswork.
6. Optical-SAR Fusion - combined products that see through cloud and at night.

The prototype is a complete running system, not a mock-up: a browser interface with an interactive 3-D globe, daily NASA satellite mosaics, eight switchable imagery layers and scientific overlays; ten real satellite sites ingested on genuine Copernicus Sentinel-2 optical imagery, six of them also carrying Sentinel-1 SAR (Assam, Delhi, Sundarbans, Punjab, Gangotri, Rondonia, a wildfire site, the Nile, Dubai, Kaziranga; the four without a SAR tile demonstrate graceful optical-only degradation); a live AI console where the agent trace animates through every processing step; Box / SHAP / LIME explainability modes; automatic historical change comparison that fetches the earliest verified scene, co-registers it with the latest and marks spectral change; and one-click PDF reports with a full hash-logged audit trail.

The same architecture carries our forward roadmap additively: native ISRO constellation ingestion (Cartosat, RISAT, Resourcesat, NISAR via Bhoonidhi and MOSDAC), voice and Indian-language queries, continuous monitoring with evidence-backed alerts, edge and offline deployment, and time-series analytics - each arriving as a new sensor profile, language pack or specialist head behind the same contract, with no rewrite of the core system.

=====================================================================
FIELD 2: HOW IT ADDRESSES THE PROBLEM
=====================================================================

SIH26167 asks for an interactive vision-language assistant that performs multimodal remote-sensing image analysis through text queries. Satellite data is abundant but answers are scarce, because three barriers stand between imagery and the people who need it. SatQuery AI is engineered against each one.

1. The expertise barrier. Extracting one fact - has this lake shrunk, is this field flooded, did built-up area expand - normally demands a GIS analyst, desktop software, index knowledge and hours of work. SatQuery AI replaces the workflow with a question: the user types it in plain language, and the agentic controller handles interpretation, model selection and parameter binding. No GIS training, no model choice, no parameter tuning.

2. The modality barrier. Optical imagery is blinded by cloud exactly when it matters most - floods, cyclones, monsoon agriculture. SatQuery AI treats SAR as a first-class citizen: Sentinel-1 tiles are despeckled, band slots abstract optical and radar into one canonical Tile contract, and fusion queries combine both so the system sees through cloud and at night. Where SAR coverage is unavailable, the system degrades gracefully to optical-only and says so in the answer.

3. The trust barrier. Generic vision-language models produce fluent prose with no evidence, no calibrated confidence and no audit trail, so their output cannot live in an operational report. Every SatQuery AI answer carries geo-anchored overlays, measured areas, per-task temperature-scaled confidence, SHAP / LIME explanations, multi-signal reconciliation and a hash-logged trace; a failed precondition produces a reasoned abstention naming the failed check rather than a guess. The system treats 'I cannot answer that reliably' as a correct answer.

The result is a system that matches the problem statement task by task: text-query interaction, multimodal imagery (optical MSI and SAR), question answering, description, grounding, change analysis and fusion - all behind one auditable pipeline, with measured results on three public remote-sensing VQA benchmarks (RSVQA-LR 80.8%, VRSBench 83.8%, CDVQA 77.2% under the supervised deployment protocol) and a working prototype on real pixels today.

=====================================================================
FIELD 3: INNOVATION AND UNIQUENESS OF THE SOLUTION
=====================================================================

1. Agentic controller, not a monolithic VLM. Interpretation, gating, specialist selection, execution and integration are separate auditable steps, each emitting its own trace line. This is what makes the system verifiable - and what makes a wrong routing detectable rather than silent.

2. Cross-signal reconciliation. Each answer is cross-examined by the independent signals behind it - a measured spectral fraction, a trained land-cover classifier and the trained answer scorer (optical vs SAR in fusion): when they disagree, the answer states the conflict and drops confidence. A single vision-language model cannot offer this, because it has no independent signals to cross-examine.

3. Resolution-gap engineering. The system is trained largely at 10 m Sentinel resolution but conditions on ground-sample distance as a first-class input across 0.3-10 m, so it is built to serve ISRO's sub-metre Cartosat and RISAT imagery - the exact regime the evaluation set targets.

4. Abstention as a capability. An unmet precondition - one image where two are required, a cloud-covered tile, an unsupported task - triggers a reasoned decline naming the failed check. Abstention precision is a reported metric, not a disclaimer.

5. Sensor-agnostic canonical tiles. Semantic band slots, GSD, CRS and quality masks make every downstream stage platform-independent. Adding Cartosat, RISAT, Resourcesat or NISAR is a sensor profile, not a code rewrite.

6. Evidence-first UX. Overlays, SHAP / LIME explanations, calibrated confidence, hash-logged traces and PDF reports are the product, not an afterthought - designed for operational reporting standards, not demos.

7. A working prototype with measured numbers. Six pipeline stages run end-to-end on real pixels: the router routes 100% of the live evaluation suite correctly, VQA validation accuracy is 76.3%, pair co-registration repair reduced residual misalignment from 4.18 px to 0.14 px, and three public benchmarks are evaluated under a supervised deployment protocol.

=====================================================================
FIELD 4: TECHNOLOGIES TO BE USED
=====================================================================

Vision-language (current build): ResNet-50 self-supervised Sentinel-2 backbone (torchgeo MoCo) with a 12-channel band projector, GSD-conditioning MLP and Sentinel-1 SAR encoder; MiniLM-L6 sentence embeddings for the question; FiLM-conditioned option-scoring VQA head; BigEarthNet-19 land-cover caption head. (RemoteCLIP / SigLIP / LLaVA-class encoders and QLoRA adaptation are roadmap reference architectures, not the shipped build.)

Specialist methods (current build): grounding and detection by spectral-index classification (NDWI / NDVI / NDBI / NBR) with connected-component regions; change detection by change-vector analysis over per-class spectral deltas; optical-SAR fusion by late fusion of the two land-cover heads. Discrete-object detection (ships, aircraft, vehicles) is deliberately out of scope and the gate abstains on it rather than guessing. (ChangeFormer / BiT, CLIPSeg + SAM and open-vocabulary detectors are roadmap reference heads.)

Geospatial: rasterio / GDAL; pyproj; refined-Lee SAR speckle filter; phase-correlation co-registration; Sentinel-2 SCL cloud and cloud-shadow masking; spectral index measurement (NDWI/NDVI class).

Backend: Python 3.12; FastAPI (/upload, /query, /report, /health); PyTorch; ONNX Runtime and int8 quantisation; LRU model registry; tile caching.

Frontend: HTML/CSS/JS; three.js 3-D globe; Leaflet with Esri World Imagery; canvas overlays; vendored assets for offline mode.

Data and evaluation: BigEarthNet.txt; VRSBench; RSVQA LR/HR; CDVQA; LEVIR-CD / OSCD / SECOND; DOTA / DIOR / LoveDa; EuroSAT; Copernicus Sentinel-1/2; ISRO / SAC evaluation pairs; Bhoonidhi and MOSDAC portals for native ISRO data integration.

=====================================================================
FIELD 5: METHODOLOGY AND PROCESS FOR IMPLEMENTATION
=====================================================================

The system is a six-stage pipeline. Each stage publishes a typed contract, so components are developed, tested, benchmarked and swapped independently.

Stage 0 - Ingestion and sensor abstraction. GeoTIFFs from any supported sensor enter a canonical Tile contract: semantic band slots, ground-sample distance, CRS, affine transform, acquisition timestamp and quality mask. Optical bands are normalised; SAR is despeckled; SCL layers drive cloud/shadow masks; bi-temporal and cross-modal pairs are checked and co-registered by phase correlation. On a real Sentinel-2 pair this repair reduced residual misalignment from 4.18 px to 0.14 px.

Stage 1 - Agentic controller. A trained six-way intent router (about 85% five-fold cross-validation, 100% on the live evaluation suite) classifies the query; slot extraction pulls targets, dates, directions and comparison phrases; a five-field compatibility gate checks image count, modality, format, metadata and task compatibility; a declarative registry filters specialists by task, modality and GSD range, with parameter binding restricted to each tool's permitted parameters. A failed gate short-circuits to a reasoned abstention that is itself a complete, scoreable trace.

Stage 2 - Specialist registry. Six heads (VQA, describe, ground, change, change-QA, fusion) share one adapted backbone. Heads are swappable, small and versioned behind a single run() contract, so a new specialist joins without touching the controller.

Stage 3 - Integration, confidence and abstention. Per-task temperature scaling calibrates confidence so '0.8' means the same across heads; multi-signal reconciliation surfaces disagreement; abstention thresholds decide answer versus decline.

Stage 4 - Trace, evidence and reporting. Every step emits a TraceStep with tool name and version, weights hash, parameters, input hash, output summary, confidence, latency and gate verdict. The same record renders the live view, the JSON API response and the PDF report.

Stage 5 - Serving and interface. FastAPI drives the browser front-end: 3-D globe with daily NASA mosaics, eight imagery layers, scientific overlays (aerosol, cloud-top temperature, sea-surface temperature, chlorophyll, snow, sea ice, active fires), automatic zoom from globe to sub-metre aerial view, six feature benches and a live agent-trace panel.

Stage 6 - Evaluation harness. Router accuracy, pipeline accuracy, abstention rate, benchmark accuracy by question type and calibration are scored continuously and written to machine-readable scorecards.

Build order: sensor profiles and tile contract first (everything depends on it), then controller / gate / registry / trace, then QLoRA adaptation with GSD conditioning, then task heads and benchmarks, then calibration / conflict / abstention, then API / GUI / report / offline packaging, and finally ablations and the full scorecard.

Forward roadmap within the same architecture: (a) native ISRO sensor profiles for Cartosat-2S/3, RISAT-1A/EOS-04, Resourcesat-2/2A and NISAR via Bhoonidhi and MOSDAC; (b) voice input/output and query understanding in Hindi and major regional languages; (c) continuous monitoring subscriptions that push evidence-backed change alerts; (d) ONNX int8 edge and offline deployment with a progressive web app; (e) multi-date time-series analytics with trend charts and anomaly detection; (f) expanded benchmarks (RSVQA-HR, caption metrics, grounding IoU, change-mask F1/IoU) and formal ECE calibration; (g) sector packs for agriculture, forestry, water, urban and disaster; (h) a versioned model registry with drift monitoring and on-prem deployment for state remote-sensing centres.

=====================================================================
FIELD 6: POTENTIAL CHALLENGES AND RISKS, AND STRATEGIES TO OVERCOME THEM
=====================================================================

1. Sensor and resolution diversity. ISRO evaluation imagery spans sub-metre Cartosat optical to RISAT SAR, while most public training data is 10 m Sentinel. Strategy: the canonical Tile contract with semantic band slots and GSD conditioning (0.3-10 m) is designed for exactly this gap; input-mode augmentation (band dropout, RGB stretch) teaches the encoder partial-band and stretched-RGB inputs; every specialist declares its supported modality and GSD range in the registry, so the gate routes around unsupported combinations instead of failing silently.

2. Cloud cover and noisy SAR. Optical imagery fails under cloud, and SAR carries speckle. Strategy: Sentinel-2 SCL cloud/shadow masking at ingest, refined-Lee despeckling for SAR, optical-SAR fusion as a first-class intent, and honest abstention when preconditions fail. Where SAR is unavailable the system degrades to optical-only and states it.

3. Query ambiguity and misrouting. Natural language is open-ended. Strategy: a trained six-way router with slot extraction, a five-field compatibility gate before execution, and a reasoned abstention that names the failed check. Router accuracy is measured continuously (100% on the live suite, about 85% cross-validation on held-out paraphrases).

4. Model hallucination and trust. Fluent prose without evidence blocks operational adoption. Strategy: evidence-first answers (overlays, measured areas, SHAP / LIME), per-task calibrated confidence, multi-signal reconciliation, hash-logged traces and PDF reports, plus abstention as a reported metric.

5. Compute and deployment constraints. District hardware is modest. Strategy: the router and calibrated heads run on CPU; QLoRA keeps training within a single consumer GPU; ONNX int8 quantisation, an LRU model registry and a tile cache keep serving light; vendored assets and a pre-warmed tile cache support offline mode.

6. Data licensing and availability. Strategy: the entire stack is built on open data (Copernicus, BigEarthNet.txt, VRSBench, RSVQA, CDVQA, LEVIR-CD, DOTA, DIOR, LoveDa) and open tooling, so there are no licence blockers; ISRO data arrives through Bhoonidhi and MOSDAC integration on the same profile mechanism.

7. Adoption by non-experts. Strategy: the interface is a question box on a familiar globe; no GIS vocabulary, indices or parameters are exposed; the roadmap adds voice and Indian-language input so a field officer can ask in their own language.

=====================================================================
FIELD 7: POTENTIAL IMPACT ON TARGET AUDIENCE
=====================================================================

Non-expert decision-makers - district officers, panchayat and municipal staff, NGO field teams, disaster responders: satellite evidence becomes a question in plain language instead of a workflow, a model choice and a parameter set. Questions that took a GIS desk hours are answered in seconds with the evidence attached.

ISRO / SAC and state remote-sensing centres: the auditable trace (task, model, version, weights hash, parameters, confidence) makes an AI answer defensible inside an operational report, which is precisely what blocks adoption of AI in official workflows today. Signal reconciliation and abstention give reviewers a way to trust - or challenge - every answer.

Analysts and scientists: routine questions are answered and evidenced automatically, so expert time shifts to the cases the system declines, which it names explicitly. The registry contract means a new specialist can be added behind the same run() signature without touching the controller.

Agriculture and water departments: crop-area change, irrigation and water-body extent answered with masks and areas, supporting scheme monitoring and drought response.

Disaster management authorities: SAR sees through cloud and at night, so flood and damage extent are available when optical satellites are useless; before/after comparisons render automatically for rapid situation reports.

Forest, environment and urban bodies: deforestation, encroachment, wetland shrinkage and built-up expansion tracked from open multitemporal imagery at query speed.

Researchers and students: one registry, one contract, six tasks - an extensible research platform with published benchmarks and a reproducible trace, not a black box.

=====================================================================
FIELD 8: BENEFITS OF THE SOLUTION (SOCIAL, ECONOMIC, ENVIRONMENTAL)
=====================================================================

Social: democratises satellite intelligence - no GIS training needed. Faster, evidence-backed disaster and flood assessment where cloud cover blocks optical imagery. Plain-language access extends Earth observation to panchayats, municipalities, NGOs and citizens, with voice and Indian-language support on the roadmap.

Economic: one shared adapted backbone replaces several siloed model deployments; analyst hours shift from routine extraction to judgement; the stack runs on CPU for routing and calibrated heads, so district-scale deployment needs no GPU workstation; open data and open weights mean no licence cost.

Environmental: continuous, low-cost monitoring of deforestation, water bodies, wetlands, encroachment and land-cover change from freely available multitemporal imagery; monitoring subscriptions on the roadmap push evidence-backed alerts when change is detected, shortening response time to environmental damage.

Strategic and trust: confidence is calibrated, conflicts between specialists are surfaced, and the system declines rather than guessing. Every answer carries a hash-logged, re-runnable trace - designed for the reporting standards of the institutions that will deploy it, and aligned with ISRO's goal of turning Earth-observation archives into actionable, auditable insight.

=====================================================================
FIELD 9: DETAILS / LINKS OF REFERENCE AND RESEARCH WORK
=====================================================================

Datasets - adaptation and training: BigEarthNet.txt (Sentinel-1 SAR + Sentinel-2 MSI + text; arxiv.org/abs/2603.29630), BigEarthNet-MM (bigearth.net), EuroSAT, LEVIR-CD, OSCD, SECOND (change detection; SECOND underlies CDVQA), DOTA, DIOR, LoveDa, ISPRS Potsdam and Vaihingen (high-resolution holdout).

Benchmarks - prescribed evaluation: VRSBench (vrsbench.github.io), RSVQA LR / HR (rsvqa.sylvainlobry.com), CDVQA (change-based VQA, scored by question type), ISRO / SAC evaluation set (pre-georeferenced, co-registered Cartosat-2S optical + RISAT SAR pairs). Metrics: accuracy by question type, CIDEr / BLEU-4 / ROUGE-L, acc@IoU 0.5, change F1 / IoU, per-class IoU, ECE, abstention precision.

Models and methods: RemoteCLIP - RS vision-language pretraining (arxiv.org/abs/2306.11029); SigLIP (2303.15343); QLoRA - 4-bit adapter tuning (2305.14314); PEFT; LLaVA - visual instruction tuning (2304.08485); GeoChat (2311.15826); ChangeFormer (2201.01293); BiT (2103.00208); CLIPSeg (2112.10003); SAM (2304.02643); temperature scaling for calibration (1706.04599).

Tooling and data portals: PyTorch, FastAPI, rasterio / GDAL, pyproj, ONNX Runtime, three.js, Leaflet; Copernicus Sentinel-1/2; NASA GIBS (VIIRS / MODIS daily mosaics); ISRO Bhoonidhi and MOSDAC portals for native constellation integration.

Prototype artefacts in the team repository: working browser GUI (3-D globe, satellite basemap, six feature benches, live agent trace), FastAPI backend with /upload /query /report /health endpoints, PDF report generator, evaluation scorecards, and a full engineering specification of the six-stage pipeline.

=====================================================================
CONDENSED VERSION - IF THE PORTAL HAS ONE COMBINED DESCRIPTION FIELD
=====================================================================

SatQuery AI is an agentic vision-language assistant that answers plain-language questions over satellite imagery. A user types a question - 'Did the flood extent grow since March?', 'Is there water near this village?', 'How much farmland became built-up last year?' - and receives an evidence-grounded answer in seconds: calibrated confidence, geo-anchored overlays, spectral measurements and a downloadable audit report.

The system replaces the GIS workflow with a conversation while keeping scientific rigour. An agentic controller (not a generic LLM) interprets the query, validates it against a five-field compatibility gate, selects the right specialist from a declarative registry, runs it on canonical sensor-agnostic tiles, and merges outputs with multi-signal reconciliation and honest abstention. Six capabilities sit behind one interface: visual question answering, scene description, geo-anchored grounding, change detection, change QA and optical-SAR fusion. Optical and SAR imagery share one canonical Tile contract with semantic band slots, GSD conditioning across 0.3-10 m, cloud/shadow masking, refined-Lee despeckling and phase-correlation co-registration, so Sentinel, Cartosat and RISAT products are handled by the same pipeline.

Trust is engineered in, not promised: every answer carries geo-anchored boxes and masks, measured class areas, temperature-scaled confidence, Box / SHAP / LIME explanations, and a hash-logged trace of every step that also renders a one-click PDF report. When the signals behind an answer disagree, the conflict is surfaced; when a precondition fails, the system declines and names the failed check - abstention is a reported metric, not a disclaimer.

The working prototype runs end-to-end on real pixels: a 3-D globe browser interface with daily NASA mosaics and scientific overlays, ten real satellite sites on genuine Copernicus Sentinel-2 optical imagery (six also with Sentinel-1 SAR), a live AI console, automatic historical change comparison with spectral-change pixels marked, and measured results - router 100% on the live evaluation suite, VQA validation 76.3%, RSVQA-LR 80.8%, VRSBench 83.8%, CDVQA 77.2% under the supervised deployment protocol, and pair co-registration repair from 4.18 px to 0.14 px residual.

Impact: district officers, disaster responders, panchayats, NGOs and analysts get satellite evidence as a plain-language question with the proof attached; ISRO / SAC and state remote-sensing centres get an auditable trace (task, model, version, weights hash, parameters, confidence) that makes an AI answer defensible in an operational report; farmers, forest and water departments get flood, deforestation, encroachment and water-extent answers at query speed, with SAR seeing through cloud when optical cannot.

The architecture carries the roadmap additively: native ISRO constellation ingestion (Cartosat-2S/3, RISAT-1A/EOS-04, Resourcesat-2/2A, NISAR via Bhoonidhi and MOSDAC), voice and Indian-language queries, continuous monitoring with evidence-backed change alerts, ONNX int8 edge and offline deployment, time-series analytics with trend charts, expanded benchmarks and calibration, sector packs for agriculture, forestry, water, urban planning and disaster management, and on-prem deployment for state remote-sensing centres. Every extension arrives as a new sensor profile, language pack or specialist head behind the same contract - growth is additive, not a rewrite.

=====================================================================
ABSTRACT / SUMMARY (UNDER 10,000 CHARACTERS)
=====================================================================

SatQuery AI is an agentic vision-language assistant that answers plain-language questions over satellite imagery. A user types a question - 'Did the flood extent grow since March?', 'Is there water near this village?', 'How much farmland became built-up last year?' - and receives an evidence-grounded answer in seconds: a calibrated confidence score, geo-anchored visual overlays, spectral measurements and a downloadable audit report. It converts satellite archives from a specialist GIS workflow into a conversation.

Earth observation is drowning in data and starving for answers. Three barriers keep satellite intelligence away from decision-makers: the expertise barrier (extracting one fact normally demands a GIS analyst, desktop software and hours of work), the modality barrier (optical imagery is blinded by cloud exactly when it matters most - floods, cyclones, monsoon agriculture - while SAR, which penetrates cloud and works at night, is understood by almost no general-purpose AI tool), and the trust barrier (generic vision-language models produce fluent prose with no evidence, no calibrated confidence and no audit trail, so their output cannot enter an operational report).

SatQuery AI is engineered against all three. An agentic controller - deliberately not a generic LLM - interprets each query, validates it against a five-field compatibility gate, selects the right specialist from a declarative registry, executes it on canonical sensor-agnostic tiles, and merges outputs with multi-signal reconciliation and honest abstention. Six capabilities live behind one interface: visual question answering, scene description, geo-anchored grounding, change detection, change QA and optical-SAR fusion. Optical and SAR imagery share one canonical Tile contract with semantic band slots, GSD conditioning across 0.3-10 m, cloud/shadow masking, refined-Lee despeckling and phase-correlation co-registration, so Sentinel, Cartosat and RISAT products are handled by the same pipeline.

Trust is engineered in, not promised. Every answer carries geo-anchored boxes and masks, measured class areas, temperature-scaled confidence, Box / SHAP / LIME explanations, and a hash-logged trace of every processing step that also renders a one-click PDF report. When the signals behind an answer disagree, the conflict is surfaced and confidence drops; when a precondition fails, the system declines and names the failed check. Abstention is a reported metric, not a disclaimer.

The working prototype runs end-to-end on real pixels: a browser interface with an interactive 3-D globe, daily NASA mosaics, scientific overlays and a live agent-trace console; ten real satellite sites on genuine Copernicus Sentinel-2 optical imagery (six also with Sentinel-1 SAR); automatic historical change comparison with spectral-change pixels marked; and measured results - the router classifies 100% of the live evaluation suite correctly, VQA validation accuracy is 76.3%, and public benchmark scorers reach RSVQA-LR 80.8%, VRSBench 83.8% and CDVQA 77.2% under the supervised deployment protocol, with pair co-registration repair reducing residual misalignment from 4.18 px to 0.14 px.

Impact is broad and concrete. District officers, panchayats, municipal staff, NGO field teams and disaster responders get satellite evidence as a plain-language question with the proof attached. ISRO / SAC and state remote-sensing centres get an auditable trace - task, model, version, weights hash, parameters, confidence - that makes an AI answer defensible in an operational report, which is precisely what blocks adoption today. Agriculture, water, forest, urban and disaster-management departments get flood extent, deforestation, encroachment, water-body shrinkage and built-up expansion answered at query speed, with SAR seeing through cloud when optical cannot. Analysts shift from routine extraction to judgement, and researchers gain an extensible platform where a new specialist joins behind a single run() contract without touching the controller.

The architecture carries the roadmap additively: native ISRO constellation ingestion (Cartosat-2S/3, RISAT-1A/EOS-04, Resourcesat-2/2A and NISAR via Bhoonidhi and MOSDAC), voice and Indian-language queries for field officers, continuous monitoring subscriptions that push evidence-backed change alerts, ONNX int8 edge and offline deployment for district hardware, multi-date time-series analytics with trend charts and anomaly detection, expanded benchmarks and formal calibration, sector packs for agriculture, forestry, water, urban planning and disaster management, and on-prem deployment for state remote-sensing centres. Every extension arrives as a new sensor profile, language pack or specialist head behind the same contract - growth is additive, not a rewrite.

SatQuery AI turns satellite archives into a conversation you can trust: it pairs the accessibility of natural-language queries with the rigour of a gated, traced, calibrated pipeline, fuses optical and radar so cloud cover stops being a deadline, and shows its work so its answers can live inside an official report. Ask the satellite. It answers, it shows, and it proves.
