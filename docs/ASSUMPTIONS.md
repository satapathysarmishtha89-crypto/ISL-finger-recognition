# ISL Fingerspelling Recognition: Dataset Assumptions & Findings
Document according to Section 3.1 of `ISL_Project_Implementation_Spec (2).md`.

---

## 1. Dataset Overview
- **Raw Image Directory:** `ISL_DATA/`
- **Total Images:** 17,292 images across 35 alphanumeric classes.
- **Classes:** 
  - 9 Numeric Digits: `1, 2, 3, 4, 5, 6, 7, 8, 9`
  - 26 Alphabet Letters: `A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z`
- **Distribution:** Approximately 400 to 690 images per class. Class `P` has the most (691 images), Class `O` has 402 images.

---

## 2. Signer & Session Identity (Section 3.1 & 3.4)
- **Session Identification:**
  - Standard files without parentheses (e.g., `0.jpg`, `1.jpg`, `100.jpg`) correspond to **Session 1 (Signer 1)** recorded in an ambient room environment.
  - Files containing `(2)` (e.g., `119 (2).jpg`) correspond to **Session 2 (Signer 2)** recorded against a studio plain white background.
  - Files containing `(3)` (e.g., `45 (3).jpg`) correspond to **Session 3 (Signer 3)** recorded against a studio plain white background.
  - For classes `O` and `W` without parenthesized names, frame index sequence blocks correspond to distinct recording sessions.
- **Signer-Independent Protocol:**
  - Training & Validation: Signer Session 1 (ambient room).
  - Held-out Test Set: Signer Sessions 2, 3, and Session 2 proxy (studio white). Zero signer overlap between train/val and test.

---

## 3. Background Environments (Section 3.1 & 3.4)
- **Background Domains Identified:**
  1. `ambient_room`: Natural indoor room background with varying ambient illumination, walls, and furniture.
  2. `studio_white`: Clean plain studio background with uniform studio lighting.
- **Background-Independent Protocol:**
  - Training & Validation: 100% ambient room domain images.
  - Held-out Test Set: 100% studio plain white domain images. Zero background overlap between train/val and test.

---

## 4. Hand Configuration (One-Handed vs. Two-Handed Signs)
- **MediaPipe Hand Detection:**
  - Single-hand detection confirmed across **17,001 out of 17,292 images (98.32% detection rate)**.
  - Indian Sign Language fingerspelling in this dataset is performed predominantly as single-handed gestures.
  - Feature representation uses normalized **42-D landmarks** (21 landmarks $\times$ 2 coordinates).
- **Fallback for Undetected Images:**
  - 291 images with extreme orientation or motion blur where landmarks were not detected are logged in `preprocessing_output/no_detection.csv`.
  - For the CNN branch, full-image resize to $224 \times 224$ is used as a fallback.
  - For the Landmark branch, confidence-aware fusion defaults to CNN-only predictions.

---

## 5. File Mapping Reference

| Spec Path | Actual Workspace Path | Description |
| :--- | :--- | :--- |
| `docs/ASSUMPTIONS.md` | `docs/ASSUMPTIONS.md` | This dataset assumptions document |
| `results/no_detection.csv` | `preprocessing_output/no_detection.csv` | List of 291 fallback images without landmarks |
| `data/landmarks/*.npz` | `preprocessing_output/landmarks_42d.npz` | 42-D normalized coordinates NumPy archive |
| `data/splits/*.csv` | `data/splits/` & `dataset_splitting_output/` | Standard, Signer, and Background split CSVs |
| `results/` | `model_1_mobilenetv2_output/` etc. | Per-model results, confusion matrices, and metrics |
