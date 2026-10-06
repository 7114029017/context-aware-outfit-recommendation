# Third-party notices

The MIT License in `LICENSE` covers only the code and documentation written for
this repository. Third-party data, code, models and publications keep their own
terms, as listed below.

## Polyvore Outfits dataset

- The experiments use the disjoint split of Polyvore Outfits (Vasileva et al.,
  ECCV 2018). `reproduction/scripts/bootstrap_data.sh` downloads the metadata
  and split files from the Hugging Face mirror `Stylique/Polyvore`.
- **Polyvore images are not redistributed due to third-party rights.** This
  repository contains no Polyvore images.
- The repository contains files derived from Polyvore Outfits: outfit and item
  identifiers, generated context-aware descriptions, CLO, MET and temperature
  estimates, W/O/S annotations, LLM-judge scores, and the precomputed image and
  text features. They are provided for research reproducibility; their use is
  subject to the terms of the Polyvore Outfits dataset.

## Upstream model code

The Hybrid Attention recommender in
`02_模型訓練和驗證_model_training_validation/main_hybrid_attention_code/`
builds on the implementation of X. Wang and Y. Zhong, "Text-Conditioned Outfit
Recommendation With Hybrid Attention Layer," *IEEE Access* 12 (2024), 281–293,
doi:10.1109/ACCESS.2023.3346933. That code is under the MIT License,
Copyright (c) 2024 WangX; see the `LICENSE` file in the same folder.

## Pretrained models (not included)

- FashionCLIP was used for the image features and the outfit-level text
  features, and SentenceBERT `distiluse-base-multilingual-cased-v2` for the item
  text features.
- Gemma-3-4B-IT generated the context-aware descriptions and the CLO and MET
  estimates. Qwen3-VL-32B-Instruct and Gemma-3-27B-IT produced the LLM-judge
  scores.

The model weights are not included. The outputs listed above are provided as
research data. Use of these models is subject to their own licenses and terms
of use.

## Publications (not included)

The reference documents used during data construction (ASHRAE
clothing-insulation references, the 2024 Adult Compendium of Physical
Activities, McIntyre (1978) and SSRN 5357611) and the upstream paper are cited
in the manuscript but are not redistributed here. See
`reproduction/docs/public_release_cleanup.md`.

## Python packages

The packages listed in `reproduction/environment/` are installed separately
and are covered by their own licenses.
