# Precomputed feature manifest

The current repository stores the following feature files through Git LFS. The SHA-256 values below are the LFS object IDs and therefore provide the expected content digest after `git lfs pull`.

| Feature file | Expected bytes | Expected SHA-256 |
|---|---:|---|
| `encoded_NewoutfitUrlTitle_en_fashionClip.pkl` | 73,590,756 | `f4d731433ef9182e16518ac7757d3aad530b362a807e0196fd79c4fa6db21ebb` |
| `encoded_category_distiluse-base-multilingual-cased-v2.pkl` | 525,651,371 | `03cb9b3d81c09f54867b00ea7c4c7bd1ccbc842555e19f757288a29f9f0537b8` |
| `encoded_no_occasion_outfitUrlTitle_en_fashionClip.pkl` | 73,590,756 | `474a6f7f1eceaeaacc8fbd173f13148385ae25380fdea1590fbc02543a1beb08` |
| `encoded_no_style_outfitUrlTitle_en_fashionClip.pkl` | 73,590,756 | `e54733fdd6a262d796b7d96749435e8c55a8ca1e2de726b0335d5b0f1ae72fcf` |
| `encoded_no_weather_outfitUrlTitle_en_fashionClip.pkl` | 73,590,756 | `f8be5098f0cd8d660522169dbd32989a76db3f6aa670ec1fb4f962ee9999783b` |
| `encoded_outfitUrlTitle_en_fashionClip.pkl` | 143,049,389 | `fb74f2c5638c175f897d07dcde5e825359127b42b330fe28783b8f50b21494a6` |
| `encoded_title_description_distiluse-base-multilingual-cased-v2.pkl` | 525,651,371 | `9e425f7e62aed5828d39cf64d4647b973d278f4579c39fccc8011c8660b201c4` |
| `img_feats_fashionClip.pkl` | 525,651,371 | `aa4859f4440e6c96b11314da30435907cf029460352318ba639a8ab41699efce` |

Run:

```bash
git lfs install
git lfs pull
python scripts/verify_feature_files.py
```

A 133–134 byte file is only an LFS pointer and must not be treated as a usable pickle.

The expected digest establishes artifact identity. It does not by itself establish which historical feature filename a 2025 run loaded; that remains a separate source-provenance question.
