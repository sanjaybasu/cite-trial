# CITE trial: code

Code for "Randomized trial of a language model verifier for AI care plans in Medicaid primary care" (ClinicalTrials.gov NCT07718893; analysis pre-registered at https://doi.org/10.17605/OSF.IO/ENXCW).

This repository holds code only. It contains no patient data, clinical notes, identifiers, adjudication sheets, or randomization keys. Aggregate results are posted at ClinicalTrials.gov.

## Contents

| Folder | What it holds |
|---|---|
| `cite_verifier/` | The CITE verifier as deployed in the trial (version cite-1.0): the five files packaged in the trial's Modal image. `verify_api.py` defines the endpoint and maps verifier output to flags and display status; `verifier_llm.py` holds the fixed verification prompt sent to Claude Opus 4.8. CITE does not retrieve guideline text at run time. |
| `analysis/` | Analysis of the registered, protocol-specified, and exploratory outcomes, adjudication packet generation, 30-day outcome linkage, and the ClinicalTrials.gov results tables. |
| `reported_numbers/` | Scripts that derive every reported number from the analysis outputs, check the reported values against them, and draw the figures. |

## Running the code

The scripts expect the authors' local directory layout and read data that are not distributed. Database access goes through an internal helper (`wm_conn`) and read-only credentials issued at run time; neither is included. The verifier needs an Anthropic API key in the environment (`CITE_MODEL` defaults to `claude-opus-4-8`); `CITE_MOCK=1` returns the response shape without a model call.
