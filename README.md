# Qwen 2.5 VL - Hallucination Detection and Mitigation

## Overview

This repository contains code for detecting and mitigating hallucinations in AI generated responses using Qwen models. It utilizes the Qwen 2.5 VL 7B model to identify hallucinations and subsequently rectify them.

## Dataset

This project uses a filtered version of the [POVID dataset](https://github.com/YiyangZhou/POVID), stored in `filtered_povid.json`. This dataset includes:
* ~1k captioning samples (2-5 sentence response) + ~1k VQA samples (1-2 sentence response)
* Each sample contains only the hallucinatory response from the original POVID dataset responses.
* In POVID, hallucinatory responses were generated by instructing ChatGPT to injecting errors into the correct (ground-truth) response. 

Our objective is to detect these errors (HAL detection) and rectify them (HAL mitigation) using Qwen models!  

## Methodology

The process involves several steps:

1.  **Claim Extraction:** Claims are extracted from (already hallucinated) responses using a Qwen LLM (specifically, Qwen 2.5 14B is currently used). This is handled within `qwen_wrapper.py`.
2.  **Hallucination Annotation:** The `qwen_HAL_annotator.py` script leverages the Qwen VL model (Qwen 2.5 VL 7B currently) to annotate each extracted claim. Annotations include:
    * **Label:** Hallucination, Non-hallucination, or Subjective.
    * **Reason:** An explanation for the annotation, aiding interpretability and error correction.
    * The annotator also performs self-checks on its annotations.
3.  **Response Rectification:** Based on the annotations, the `qwen_HAL_annotator.py` script attempts to rectify the errors found in the original response.

**Analysis:** The `analyse_qwen_annotations.py` script can be used to visualize the annotation results.

### Key Files

* `filtered_povid.json`: A sub-set of POVID dataset, containing hallucinatory samples.
* `qwen_wrapper.py`: Contains wrappers for the Qwen 2.5 LLM and Qwen 2.5 VL models.
* `qwen_HAL_annotator.py`: Core script for claim extraction, annotation, and rectification.
* `analyse_qwen_annotations.py`: Script for visualizing annotation results.
* `Qwen_HAL_Annotations.json`: Output annotations for 5 captioning and 5 VQA samples. For your reference!

## Current Status

**Work in Progress:** This repository is under active development. The quality of annotations produced by the Qwen models needs further improvement.
