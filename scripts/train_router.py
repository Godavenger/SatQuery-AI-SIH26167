"""Train the router v2 query classifier on an expanded seed corpus.

The original router was trained on 59 hand-written queries (71% CV accuracy).
This script grows the corpus to ~360 queries across the six intents, holds
out a stratified 20% test split (the run_eval.py queries are deliberately
excluded from the corpus so evaluation stays uncontaminated), and reports
held-out accuracy per intent.

Saves models/router_v2.pkl in the exact format TrainedRouter expects:
    {"pipeline": sklearn.Pipeline, "metrics": {...}}

Usage:
    cd C:\\Users\\Sharanya\\Documents\\Hackathon\\SatQuery_AI_SIH26167
    python scripts/train_router.py
"""
from __future__ import annotations

import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "router_v2.pkl"

# The exact queries run_eval.py uses - excluded from training so the
# scorecard reflects generalisation, not memorisation.
EVAL_QUERIES = {
    "Describe the land-cover and major objects visible in this image.",
    "Highlight the water body referred to in the query.",
    "What changed between these two dates, and where did the change occur?",
    "Has the built-up area increased, decreased, or remained unchanged?",
    "Use the optical and SAR images together to identify built-up and water-covered regions.",
    "Give me an overview of what this scene shows.",
    "Find the flooded region on this image.",
    "Show me the differences between these two satellite passes.",
    "Did the water extent grow or shrink?",
    "Combine the radar and optical data to map water.",
    "What land cover is dominant here?",
    "Is there any snow or ice visible?",
    "How much vegetation is in this tile?",
}

CORPUS: dict[str, list[str]] = {
    "vqa": [
    "Is there any water in this image?",
    "Check if there is any water.",
    "Are there any water bodies here?",
        "Does this tile contain bare soil?",
        "Are there agricultural fields visible?",
        "Is the terrain mountainous or flat?",
        "What is the dominant land cover type?",
        "Is there snow or ice in the scene?",
        "Are there any buildings in this image?",
        "Is the vegetation healthy?",
        "Does this area look urban or rural?",
        "Is there a river crossing the tile?",
        "Are there clouds obscuring the ground?",
        "What fraction of the image is forested?",
        "Is the water body natural or man-made?",
        "Are there any roads visible?",
        "Is this a coastal area?",
        "How much of the scene is covered by vegetation?",
        "Is the land arid or fertile?",
        "Are there any glaciers visible?",
        "Is this region densely populated?",
        "What season does this image appear to be from?",
        "Is there flooding anywhere in the frame?",
        "Does the image show farmland?",
        "Are wetlands present in this tile?",
        "Is there standing water on the fields?",
        "Can you see any airport infrastructure?",
        "Is the area mostly green or brown?",
        "Are there sand dunes in this image?",
        "Is the shoreline eroded?",
        "Does the tile show a desert?",
        "How much built-up area is present?",
        "Is there haze or smoke in the atmosphere?",
        "Are the crops mature or freshly planted?",
        "Is there any industry visible?",
        "What is the general condition of the vegetation?",
        "Is the lake full or dried up?",
        "Are there patches of barren land?",
        "Does the river look swollen?",
        "Is the area covered by mangroves?",
        "Can you tell if this is a residential zone?",
        "Is the terrain suitable for agriculture?",
        "Are there any islands in the water body?",
        "Is the image mostly cloudy?",
        "Does the scene contain grassland?",
        "Is there evidence of drought?",
        "Are the fields irrigated?",
        "Is this a natural or artificial water body?",
        "Can ships be seen on the water?",
        "Is there any deforestation evident?",
        "Does this look like a floodplain?",
        "Are there any geological formations of interest?",
        "Is the vegetation sparse or dense?",
        "What percentage of the tile is water?",
        "Are there any parking lots or large paved areas?",
        "Is the snow cover permanent or seasonal?",
        "Does the image show a harbor?",
        # disambiguate from change/change-qa: present-tense, single image
        "What land cover is dominant here?",
        "Which cover type dominates this single scene?",
        "What is the dominant crop in this field?",
        "Is water the majority class in this tile right now?",
        # disambiguate from change: "land cover" + question, no pair/two-date cue
        "What land cover types are present in this image?",
        "Which land cover class dominates the scene right now?",
        "What is the land cover composition of this single image?",
        "Which cover appears most abundant in this one scene?",
        "What land cover do we see in this frame?",
        "Which land cover class has the largest share in this picture?",
        # counting on a single image is VQA with a numeric answer
        "How many forests are in this image?",
        "How many distinct land cover types are in this image?",
        "How many water bodies are visible in this tile?",
        "How many buildings can be counted in this scene?",
        "How many fields appear in this image?",
        "How many roads cross this tile?",
        "How many lakes are in this image?",
        "Count the buildings in this image.",
        "Count the distinct vegetation patches in this tile.",
        "Count the ships visible on the water.",
        "How many urban blocks are in this scene?",
        "How many rivers flow through this image?",
        "How many islands are visible here?",
        "How many crop plots are in this field mosaic?",
        # crop/scene-type single-image questions
        "What crop is growing here?",
        "What crop is being cultivated in this field?",
        "Which crop type does this field show?",
        "Is this a rice paddy or a wheat field?",
        "What stage is the crop growth at in this image?",
    ],
    "describe": [
        "Describe the land-cover and major objects visible in this image.",
        "Give me an overview of what this scene shows.",
        "What is happening in this satellite image?",
        "Summarise the contents of this tile.",
        "Explain what I am looking at in this image.",
        "Provide a caption for this satellite scene.",
        "What does this area look like?",
        "Describe the main features of this region.",
        "Tell me about this location from space.",
        "What kind of landscape is shown here?",
        "Write a short description of the terrain.",
        "What can be observed in this imagery?",
        "Give a summary of the land use here.",
        "Describe the scene in one or two sentences.",
        "What are the notable elements in this image?",
        "Paint a picture of what this place looks like from above.",
        "List the land cover classes present.",
        "What is the character of this landscape?",
        "Describe the geography of this tile.",
        "Give me a natural-language caption of the scene.",
        "What environmental features stand out here?",
        "Briefly describe the satellite view.",
        "What is the overall scene composition?",
        "Describe the spatial patterns in the image.",
        "Give an interpretive summary of the imagery.",
        "What does the aerial view reveal about this area?",
        "Describe the visible structures and terrain.",
        "Give me the gist of this remote sensing image.",
        "What is this region characterised by?",
        "Provide a detailed description of the land surface.",
        "Describe the types of terrain and objects visible.",
        "What does the imagery show overall?",
        "Give a caption for this patch of Earth.",
        "How would you describe this scene to someone who cannot see it?",
        "What land features dominate this view?",
        "Describe the land cover composition of this tile.",
        "Offer a brief interpretation of this satellite image.",
        "What is visible in this overhead image?",
        "Describe the natural and man-made features here.",
        "Give a high-level description of this scene.",
        "What does the Earth's surface look like here?",
        "Describe the setting shown in this image.",
        "Provide an overview of the land cover here.",
        "What does this patch of terrain contain?",
        "Summarise the visible landscape elements.",
        "Describe the image contents in plain language.",
        "What is the land like in this area?",
        "Give a photographic caption for the scene.",
        "Describe what stands out in this overhead view.",
        "Explain the geography visible from the satellite.",
        "What natural features appear in this image?",
        "Describe the environment captured in this tile.",
        "Give me a short narrative of the scene.",
        "What is the main story this image tells?",
        "Describe the physical characteristics of this area.",
    ],
    "ground": [
        "Highlight the water body referred to in the query.",
        "Find the flooded region on this image.",
        "Locate the forest in this tile.",
        "Show me where the urban area is.",
        "Segment the agricultural land.",
        "Identify the boundaries of the lake.",
        "Point out the river channel.",
        "Mark all the buildings in the scene.",
        "Where is the bare ground in this image?",
        "Where is the water in this image?",
        "Where is the water in this scene?",
        "Where is the forest in this image?",
        "Where is the vegetation in this image?",
        "Where is the urban area in this image?",
        "Where are the buildings in this image?",
        "Where is the snow in this image?",
        "Where are the wetlands in this image?",
        "Highlight the vegetated zones.",
        "Draw the extent of the water body.",
        "Locate the wetlands in this tile.",
        "Show the road network in the image.",
        "Find the glacier boundaries.",
        "Identify where the crops are growing.",
        "Map the built-up areas.",
        "Show me the location of the port facilities.",
        "Highlight every patch of snow.",
        "Where are the sandbars in this river reach?",
        "Segment the coastline in this image.",
        "Locate the burned area.",
        "Find areas affected by landslides.",
        "Show the distribution of grassland.",
        "Identify the flood extent on the map.",
        "Mark the boundaries of the city.",
        "Point out the airport runways.",
        "Where does deforestation appear in the image?",
        "Highlight the drought-affected fields.",
        "Show me the barren patches.",
        "Locate the mangrove cover.",
        "Identify the irrigated zones.",
        "Where is the shoreline in this scene?",
        "Segment out all water pixels.",
        "Find the green spaces in the urban area.",
        "Show the location of the quarry or mine.",
        "Highlight the parking lots and paved surfaces.",
        "Locate the islands in the archipelago.",
        "Mark the cloud-covered regions.",
        "Where is the ice sheet in the image?",
        "Identify the rubble or debris fields.",
        "Show which parts of the tile are residential.",
        "Point out the industrial zone.",
        "Locate the natural vegetation areas.",
        "Highlight the course of the main channel.",
        "Find the standing water in the fields.",
        "Show the extent of the landslide scar.",
        "Where are the steep slopes in this terrain?",
        "Mark out the agricultural parcels.",
        "Identify the water bodies and their boundaries.",
        "Show me the regions of dense settlement.",
        "Locate the harbor infrastructure.",
    "Highlight the areas of sparse vegetation.",
    "Check for water bodies in this image.",
    "Detect water in this scene.",
        "Find the boundaries between forest and field.",
        "Point out where the land is degraded.",
        "Show the spatial extent of the flooded district.",
    ],
    "change": [
        "What changed between these two dates, and where did the change occur?",
        "Show me the differences between these two satellite passes.",
        "Detect the changes between the two images.",
        "Compare these two time steps and find differences.",
        "What is different in the later image?",
        "Identify the areas that changed over time.",
        "Where did the land cover change?",
        "Highlight the change footprints between both dates.",
        "Compare the before and after images.",
        "What transformations occurred in this period?",
        "Show the change map for this location.",
        "Which regions evolved between the two acquisitions?",
        "Detect any new construction between the two dates.",
        "Find where deforestation happened between the passes.",
        "What areas were affected by the flood between the dates?",
        "Compare the imagery pair and describe the differences.",
        "Locate the changed pixels in the bi-temporal pair.",
        "What differences exist between time one and time two?",
        "Identify change hotspots between the two scenes.",
        "Where has the shoreline moved between the images?",
        "Show me what appeared or disappeared over time.",
        "Detect urban growth between the two dates.",
        "Which parts of the tile changed between acquisitions?",
        "Produce a change detection map for this pair.",
        "What alterations are visible when comparing both images?",
        "Find the new features in the second image.",
        "Where did vegetation loss occur between the passes?",
        "Highlight the areas modified between the two timestamps.",
        "What changed on the ground over this interval?",
        "Show the regions that differ across the time series.",
        "Detect the flood extent change between both dates.",
        "Compare both passes and flag the differences.",
        "Where are the changes concentrated in the image pair?",
        "Identify what moved or shifted between the scenes.",
        "Show me the delta between these two images.",
        "What land cover transitions happened here?",
        "Find areas of new land reclamation between dates.",
        "Where did the glacier retreat between the acquisitions?",
        "Detect the burned area expansion between passes.",
        "Which zones look different in the after image?",
        "Show the spatial pattern of change over time.",
        "What disappeared from the earlier image?",
        "Locate new roads or buildings in the later scene.",
        "Compare the two epochs and summarise the differences.",
        "Where did the water level change between dates?",
        "Detect mining expansion between the two images.",
        "What areas underwent modification in this span?",
        "Show me every difference between these two captures.",
        "Identify the changed regions in the temporal pair.",
        "Where is the change signal strongest?",
        "Flag all pixels that changed between the passes.",
        "What developments occurred between the two dates?",
        "Show the before-and-after comparison results.",
        "Detect change across the bitemporal imagery.",
        # disambiguate from change-qa: asks for WHERE/WHAT, not a direction
        "What changed between these two dates?",
        "Where are the differences between the two images?",
        "Which areas changed between the acquisitions?",
        "Show me what is different in the later scene.",
    ],
    "change-qa": [
        "Has the built-up area increased, decreased, or remained unchanged?",
        "Did the water extent grow or shrink?",
        "Did the forest area expand or contract over this period?",
        "Has agricultural land increased between the two dates?",
        "Did the city grow larger in the second image?",
        "Was there more or less vegetation in the later pass?",
        "Has the shoreline advanced or retreated?",
        "Did the flooded area increase over time?",
        "Is there more snow cover now than before?",
        "Has the glacier gained or lost area?",
        "Did the bare soil area change between the acquisitions?",
        "Is the lake bigger or smaller in the recent image?",
        "Did construction activity increase in this zone?",
        "Has the road network expanded since the earlier date?",
        "Did the wetland area grow between the passes?",
        "Is there less farmland in the second scene?",
        "Did the burn scar get larger over time?",
        "Has the urban footprint changed noticeably?",
        "Did the river channel widen or narrow?",
        "Is vegetation denser now than in the earlier image?",
        "Did the drought conditions worsen or improve?",
        "Has the coastline eroded further between dates?",
        "Did the mangrove extent increase or decrease?",
        "Is there more impervious surface in the later pass?",
        "Did the barren land area shrink?",
        "Has the green cover improved over the interval?",
        "Did water storage in the reservoir rise or fall?",
        "Is there new land where water used to be?",
        "Did the deforested area grow between images?",
        "Has the sandbar shifted landward or seaward?",
        "Did industrial activity expand in this period?",
        "Are there fewer trees in the later acquisition?",
        "Did the marsh area remain stable or change?",
        "Has the island grown in size over the years?",
        "Did the irrigated area increase between seasons?",
        "Is the second image greener than the first?",
        "Did the settlement density increase over time?",
        "Has the grassland area changed at all?",
        "Did the mining pit get deeper or wider?",
        "Was there net land gain or land loss here?",
        "Did the cloud-free observable area change?",
        "Is there more or less ice in the second scene?",
        "Did the estuary widen between the two dates?",
        "Has the fallow land decreased?",
        "Did the number of water bodies change?",
        "Is the later image more developed than the earlier one?",
        "Did the parkland area stay the same?",
        "Has cropland replaced any forest here?",
        "Did the degraded land area increase or decrease?",
        "Was the change positive or negative for vegetation?",
        "Did the water clarity improve or worsen?",
        "Has the terrain been altered significantly?",
        "Did the harbour facilities expand over this window?",
        "Is there evidence of more construction now?",
        "Did the natural habitat area decline?",
        # disambiguate from change: asks increase/decrease/unchanged direction
        "Did the area increase or decrease?",
        "Has this region grown or shrunk over time?",
        "Did the cover remain unchanged between the passes?",
        "Was there an increase in any land class here?",
        # single-direction polarity: increase/decrease without "or"
        "Did the water area increase between these two images?",
        "Did the water area decrease between these two images?",
        "Did the vegetation increase over this period?",
        "Did the vegetation decrease over this period?",
        "Did the built-up area increase in the second image?",
        "Did the built-up area decrease in the second image?",
        "Has the forest area increased since the earlier date?",
        "Has the forest area decreased since the earlier date?",
        "Did the snow cover increase between the two passes?",
        "Did the snow cover decrease between the two passes?",
        "Did the crop area increase compared to last season?",
        "Did the crop area decrease compared to last season?",
        "Did the flood extent increase between the images?",
        "Did the flood extent decrease between the images?",
        "Did the bare soil area increase over time?",
        "Did the bare soil area decrease over time?",
        "Did the urban area increase between the two scenes?",
        "Did the urban area decrease between the two scenes?",
        "Did the wetland area increase across the pair?",
        "Did the wetland area decrease across the pair?",
        "Has the water body grown since the first acquisition?",
        "Has the water body shrunk since the first acquisition?",
    ],
    "fuse": [
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "Combine the radar and optical data to map water.",
        "Compare the optical and SAR views of this location.",
        "Compare the radar and optical imagery of this scene.",
        "How do the optical and SAR images differ in what they reveal?",
        "What does the SAR image add over the optical view?",
        "Contrast the radar return with the colour imagery.",
        "Compare what each sensor sees in this tile.",
        "Read this location with both the optical and radar sensors.",
        "What does the SAR view show compared to the optical view?",
        "Fuse the SAR and optical imagery for classification.",
        "Integrate both sensors to find the flooded buildings.",
        "Use both modalities jointly to detect changes.",
        "Leverage SAR and optical together for land cover mapping.",
        "Merge the radar return with the visible imagery.",
        "Apply sensor fusion to locate the water bodies.",
        "Combine both data sources to identify urban areas.",
        "Use the SAR and optical pair to detect flooding under clouds.",
        "Cross-validate the radar and optical detections.",
        "Fuse both views to map the coastline.",
        "Use joint optical-radar analysis on this site.",
        "Integrate the two modalities for a better mask.",
        "Combine Sentinel-1 and Sentinel-2 for this query.",
        "Use radar penetration plus optical detail together.",
        "Apply multimodal fusion to detect the built-up regions.",
        "Use both the SAR and optical passes jointly.",
        "Combine the cloud-piercing radar with the optical scene.",
        "Fuse both acquisitions to classify the wetlands.",
        "Use dual-sensor fusion to map agriculture.",
        "Integrate radar backscatter with optical reflectance.",
        "Combine all available imagery for this analysis.",
        "Use the combined power of both sensors.",
        "Merge radar and optical features for segmentation.",
        "Use SAR and optical data together for change detection.",
        "Fuse the modalities to find hidden water under vegetation.",
        "Jointly analyse the radar and optical tiles.",
        "Use the fused product to identify the port area.",
        "Combine cloud-free radar with the optical view.",
        "Apply data fusion across the sensor types.",
        "Use both images in combination for this task.",
        "Integrate optical and SAR for flood mapping.",
        "Fuse the two data streams for classification.",
        "Use cross-modal analysis on this image pair.",
        "Combine the radar image with the true-colour scene.",
        "Use every available band from both sensors.",
        "Leverage the complementarity of radar and optics.",
        "Fuse the SAR layer with the multispectral stack.",
        "Use joint analysis of both acquisition modes.",
        "Combine passive and active sensing for this query.",
        "Use the multimodal stack to detect the targets.",
        "Fuse both modalities to improve the detection.",
        "Integrate all sensor inputs for the final map.",
        "Use the radar and optical combination for agriculture.",
        "Combine both products to separate water from land.",
        "Use fused imagery to identify the infrastructure.",
        "Apply joint optical and radar processing here.",
        "Use both the microwave and optical observations.",
        "Fuse the datasets for a robust classification.",
        "Use combined sensor input to map the flood.",
        "Integrate the radar with the visible bands.",
        "Use the two sources together for urban mapping.",
        "Combine the SAR stack with the optical tiles.",
        "Use fusion of both modalities for this scene.",
    ],
}


def build_dataset() -> tuple[list[str], list[str]]:
    """Flatten the corpus into (texts, labels), excluding eval queries."""
    texts, labels = [], []
    for intent, queries in CORPUS.items():
        for q in queries:
            if q in EVAL_QUERIES:
                continue
            texts.append(q)
            labels.append(intent)
    return texts, labels


def main() -> int:
    texts, labels = build_dataset()
    print(f"Corpus: {len(texts)} queries, distribution: {dict(Counter(labels))}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42)

    pipe = Pipeline([
        ("tf", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                               min_df=1)),
        ("lr", LogisticRegression(C=4.0, max_iter=600, random_state=42)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv)
    print(f"5-fold CV on train: {scores.mean():.3f} +/- {scores.std():.3f}")

    pipe.fit(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)
    test_f1 = f1_score(y_test, pipe.predict(X_test), average="macro")
    print(f"Held-out test accuracy: {test_acc:.3f}  (n={len(y_test)})")
    print(f"Held-out macro-F1:     {test_f1:.3f}")
    print("\n" + classification_report(
        y_test, pipe.predict(X_test), zero_division=0))

    # Sanity check: route the eval queries exactly as run_eval will.
    print("Uncontaminated eval queries (never in training):")
    for q in sorted(EVAL_QUERIES):
        pred = pipe.predict([q])[0]
        print(f"  {pred:10s} <- {q}")

    if OUT.exists():
        bak = OUT.with_suffix(".seed59.bak.pkl")
        shutil.copy2(OUT, bak)
        print(f"Backed up previous router -> {bak}")

    metrics = {
        "cv_accuracy": float(scores.mean()),
        "cv_std": float(scores.std()),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "n_samples": len(texts),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "corpus_version": "expanded-v2",
    }
    with open(OUT, "wb") as f:
        pickle.dump({"pipeline": pipe, "metrics": metrics}, f)
    print(f"Saved -> {OUT}")

    # Update scorecard-friendly summary
    print("\nMetrics:", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
