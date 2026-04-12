# LLMAG-generalization-copper

Code for the LLMAG framework for geological generalization and copper occurrence integration.

## Overview

This repository contains the clean notebook implementation used for the LLMAG workflow. The framework separates:

1. **Semantic harmonization** of geological unit descriptions into controlled generalized labels.
2. **Deterministic GIS processing** for tectonic-guided polygon overlay, dissolve, cleaning, and export.
3. **Copper occurrence integration and clustering** using point overlay and Gaussian Mixture Modelling.

The LLM component is optional and is used only for semantic normalization support. All spatial processing and output generation remain deterministic.

## Main workflow

The notebook runs the following steps:

1. Read structured geological inputs.
2. Build the control mapping table.
3. Optionally normalize auxiliary semantic text with DeepSeek.
4. Merge semantic support with the control mapping.
5. Run tectonic-constrained GIS generalization.
6. Cluster copper occurrences and export outputs.

## Main input files expected by the notebook

The current notebook expects the following inputs:

- `GEOLOGY_SHP`: Pakistan geology shapefile
- `CONTROL_CSV`: unit-to-category control mapping table
- `COPPER_XLSX`: copper occurrence spreadsheet
- `FAULTS_SHP`: active fault / structural layer
- `TECTONIC_SHP`: tectonic unit polygon layer

These paths are currently set to Kaggle input locations in the notebook. If you run the workflow outside Kaggle, update the input paths accordingly.

## Optional environment variables

The notebook supports the following optional environment variables:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`
- `LEGEND_OCR_TXT_PATH`

If no API key or auxiliary text is supplied, the workflow automatically falls back to the control table and still runs deterministically.

## Main outputs produced

The notebook writes outputs into working directories and exports files such as:

- `unit_to_generalized_class_mapping.csv`
- `unit_to_generalized_class_mapping.xlsx`
- `mapping_run_metadata.json`
- `generalized_geology.gpkg`
- `generalized_geology.geojson`
- `copper_points_clustered.gpkg`
- `Mapping_summary.csv`
- `Tectonic_area.csv`

## Repository contents

Recommended repository contents:

- `llmag-agent-based-generalization.ipynb`
- `README.md`
- `requirements.txt`
- `.gitignore`
- `DATA_SOURCES.md`
- `RIGHTS_NOTICE.txt`
- `CITATION.cff`

## Installation

A typical setup is:

```bash
pip install -r requirements.txt
```

## Reproducibility note

This repository provides the clean implementation of the LLMAG workflow. Some input datasets may be restricted, licensed, private, or too large to redistribute directly in this repository. Their status and access route should be described in `DATA_SOURCES.md` and in the manuscript's Data Availability Statement.

## Suggested execution

Open the notebook and run all cells after confirming that:

- the input file paths are correct,
- required Python packages are installed,
- optional API credentials are configured only if semantic normalization is needed.

## Citation

Please cite the associated paper and archived repository release when available.
