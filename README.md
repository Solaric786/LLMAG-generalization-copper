# LLMAG Generalization and Copper Occurrence Integration

This repository contains the implementation and validation code for the revised LLMAG workflow presented in:

**LLM-Assisted Workflow for Geological Unit Harmonization and Tectonic-Unit-Constrained Map Generalization: Copper Occurrence Overlay in the Chagai Belt, Pakistan**

## Overview

LLMAG separates the workflow into three modules:

1. **LLM-assisted semantic harmonization**  
   Heterogeneous geological unit names and descriptions are mapped to a controlled age-based schema. The LLM output is treated only as a candidate mapping and is subjected to deterministic validation, source-grounding checks, CSV fallback, and documented review.

2. **Deterministic GIS generalization**  
   Geological polygons are partitioned by tectonic units and processed using fixed dissolve, topology-cleaning, clipping, and export rules. The LLM does not modify polygon geometry.

3. **Copper-occurrence overlay and descriptive clustering**  
   Copper occurrences are linked to generalized polygons using point-in-polygon overlay. Coordinate-based Gaussian Mixture Model clustering is used only to summarize spatial grouping and is not interpreted as mineral prospectivity, deposit type, or geological confidence.

## Main notebook

- `llmag-agent-based-generalization.ipynb`  
  Main end-to-end LLMAG notebook for semantic harmonization, deterministic geological generalization, copper-occurrence integration, and output generation.

## Major-revision validation scripts

The following scripts were added during the major revision to address reviewer comments and strengthen reproducibility.

### 1. Repeated-call consistency analysis

- `reviewer2_comment1_repeated_call_test.py`

This script evaluates ten independent API calls using the same archived prompt, semantic input, model alias, and request settings. It reports:

- API-call success
- JSON parsing validity
- schema compliance
- returned abbreviations and labels
- source-grounding results
- duplicate records
- accepted-label consistency
- agreement with the frozen final reviewed mapping

The repeated-call test is used only for validation and does not overwrite the final geological-unit mapping.

### 2. Matched semantic and GIS baseline comparison

- `Comment4_GIS_Baseline_Comparison_Method.py`

This script compares four semantic inputs under identical downstream GIS settings:

1. original geological unit codes
2. deterministic keyword/regular-expression mapping
3. manually curated lookup table
4. final reviewed LLMAG mapping

The comparison reports:

- Unassigned records
- disagreement with the final reviewed mapping
- output feature count
- polygon-part count
- vertex count
- boundary length
- copper-point assignment
- invalid or empty geometries
- measured computational runtime

### 3. GMM model selection and stability analysis

- `Comment5_GMM_Model_Selection.py`

This script evaluates candidate Gaussian Mixture Models for \(K=1\) to \(K=10\) using:

- AIC
- BIC
- minimum component size
- singleton-component occurrence
- repeated-seed assignment stability
- adjusted Rand index

The final descriptive model uses \(K=3\) and contains clusters A-C with counts 12, 21, and 23.

## Main workflow

The notebook performs the following operations:

1. Read geological, tectonic, structural, and copper-occurrence inputs.
2. Build or load the controlled unit-to-category mapping.
3. Optionally request candidate semantic labels from the DeepSeek API.
4. Apply deterministic schema and source-grounding validation.
5. Merge validated candidates with the control mapping using documented fallback rules.
6. Partition geology by tectonic units.
7. Dissolve polygons using the controlled geological labels.
8. Perform topology cleaning and regional clipping.
9. Link copper occurrences to generalized polygons.
10. Fit and summarize the final descriptive \(K=3\) GMM.
11. Export maps, tables, spatial layers, and audit outputs.

## Expected input files

The notebook expects paths for:

- Pakistan geology shapefile
- geological unit-to-category control mapping
- copper-occurrence spreadsheet
- active-fault or structural layer
- tectonic-unit polygon layer
- optional OCR-extracted geological legend text

The current notebook uses Kaggle input paths. Update these paths when running elsewhere.

## Optional environment variables

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`
- `LEGEND_OCR_TXT_PATH`

If no API key or OCR text is supplied, the workflow falls back to the existing control mapping and continues with deterministic GIS processing.

## Main outputs

Typical outputs include:

- final unit-to-generalized-class mapping
- semantic audit tables
- mapping run metadata
- generalized geology in GeoPackage and GeoJSON formats
- copper-point assignments
- GMM model-selection and cluster summaries
- matched-baseline comparison tables
- repeated-call consistency outputs
- publication-ready figures

## Reproducibility

The revised workflow distinguishes between:

- raw LLM candidate outputs
- deterministically accepted records
- CSV fallback records
- final reviewed mappings
- deterministic GIS outputs

The final reviewed geological-unit mapping contains 165 source-unit records. The complete source-evidence audit is not treated as an independent expert-labelled benchmark; therefore, circular accuracy, macro-F1, and confusion-matrix statistics are not reported.

The repeated-call, baseline-comparison, and GMM-validation scripts are included to reproduce the analyses added during major revision.

## Installation

```bash
pip install -r requirements.txt
```

## Suggested execution

1. Confirm all input paths.
2. Install the required Python packages.
3. Configure optional API credentials only when semantic candidate generation is required.
4. Run the main notebook.
5. Run the three validation scripts when reproducing the major-revision analyses.

## Data availability

Input datasets, licensing information, and access routes are documented in:

- `DATA_SOURCES.md`
- the manuscript Data and Code Availability section
- the associated Zenodo data and software records

## Repository contents

- `llmag-agent-based-generalization.ipynb`
- `reviewer2_comment1_repeated_call_test.py`
- `Comment4_GIS_Baseline_Comparison_Method.py`
- `Comment5_GMM_Model_Selection.py`
- `README.md`
- `requirements.txt`
- `DATA_SOURCES.md`
- `RIGHTS_NOTICE.txt`
- `CITATION.cff`

## Citation

Please cite the associated manuscript and archived Zenodo software release when using this workflow.
