# ISL Fingerspelling Recognition: Implementation Spec

> **For the coding agent (Antigravity):** Build this project end to end by following the sections in order. Do not skip the evaluation protocol in Section 5; it is the main contribution of the project. If anything is ambiguous, state your assumption in `docs/ASSUMPTIONS.md` and continue.

---

## 1. Goal

Build an **Indian Sign Language (ISL) fingerspelling recognizer** (35 classes, about 14,000 images in `ISL_DATA`) that combines:

- an image CNN branch (MobileNetV2 and EfficientNet-B0),
- a MediaPipe landmark branch (42-D DNN),
- a **hybrid late-fusion** model,

and that is evaluated **more rigorously than prior work**: standard split, **signer-independent** split and **background-independent** split. It also provides **explainability** (Grad-CAM, t-SNE) and **Gemini-based sign guidance**.

### What the literature tells us (and why this design)

| Finding from the 14 reviewed papers | Design decision |
|---|---|
| Random-split results are saturated: 99.69% (MobileNetV2, 35-class ISL, Ingoley 2025), 99.81% (ConvNeXt, ASL, Nath 2026). Another paper reports 100% CNN test accuracy (Sharma 2024), which suggests leakage or an easy split. | Accuracy on a random split cannot be the selling point. The project is differentiated by **stricter evaluation, robustness and interpretability**. |
| Signer-independent evaluation causes large drops (Kirandevraj 2025: 4.87% to 16.8% CER). Almost no other paper reports it. | Make **signer-independent and background-independent** results the headline numbers. |
| Overfitting and leakage signs (Bhagyawant 2025: 100% validation accuracy with 84% training accuracy). | Split by **group** (signer or session), never by individual frame. Keep a separate validation set. Test set is untouched until the end. |
| MediaPipe cropping plus MobileNetV2 works well and gives a small model (Ingoley 2025: 42.8 MB). | Use **MediaPipe-based hand crop** as input to the CNN branch. Report model size and latency. |
| Augmentation applied to training only (Nath 2026), and background or complex-scene handling is a known weakness (Pariwat 2021, Vu 2021). | Train-only augmentation, **including background augmentation**, to support the background-independent claim. |
| t-SNE, Cohen's Kappa and MCC were used for analysis (Nath 2026). | Report **Kappa, MCC, per-class F1 and confusion pairs** in addition to Accuracy, Precision, Recall and F1. |
| Few papers combine image and landmark features. | Hybrid late fusion plus **ablations** (CNN-only, landmark-only, fusion strategies) is the core modeling contribution. |

---

## 2. Environment and Conventions

- Language: Python 3.10+.
- Framework: **use the framework already used in my existing project code if it is present; otherwise TensorFlow/Keras**.
- Libraries: `mediapipe`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pyyaml`, `streamlit` (demo), `google-genai` (Gemini).
- Reproducibility: fixed seed (`42`) for numpy, python and the framework; all hyperparameters in `configs/config.yaml`; save every run's metrics as JSON in `results/`.
- Secrets: the Gemini API key is read from the environment variable `GEMINI_API_KEY`. **Never hard-code it.**

### Project structure

```
isl-fingerspelling/
├── configs/config.yaml
├── data/
│   ├── ISL_DATA/                 # raw images (class folders)
│   ├── landmarks/                # cached 42-D features (.npz)
│   ├── crops/                    # MediaPipe-cropped images
│   └── splits/                   # standard.csv, signer_independent.csv, background_independent.csv
├── src/
│   ├── data/        (inspect.py, extract_landmarks.py, crop_hands.py, make_splits.py, datasets.py)
│   ├── models/      (cnn.py, landmark_dnn.py, fusion.py)
│   ├── train.py
│   ├── evaluate.py
│   ├── explain/     (gradcam.py, tsne_plot.py)
│   ├── guidance/    (sign_descriptions.json, gemini_guidance.py)
│   └── utils/       (metrics.py, seed.py, plots.py)
├── app/demo.py                   # Streamlit / webcam demo
├── results/                      # metrics JSON, confusion matrices, plots
├── docs/ASSUMPTIONS.md
└── README.md
```

---

## 3. Step 1: Data Inspection and Preprocessing

### 3.1 Inspect the dataset (do this first)

Write `src/data/inspect.py` to report:

- number of classes (expect 35) and images per class (expect about 400, so about 14,000 total),
- image sizes and formats, corrupt or unreadable files,
- **whether signer or session identity is available** (folder names, filename patterns, metadata),
- **whether background information is available** (and how backgrounds differ),
- **whether signs use one hand or two hands** (important for Section 3.3).

Write findings to `docs/ASSUMPTIONS.md`. Sections 3.4 and 5 depend on this.

### 3.2 Image branch preprocessing

1. Run MediaPipe Hands on each image.
2. Crop a square bounding box around the hand with a small margin. Do not clip the hand contour. Pad to square, then resize to **224x224**.
3. Normalize using the preprocessing function that matches the backbone (`mobilenet_v2.preprocess_input`, `efficientnet.preprocess_input`).
4. If no hand is detected, fall back to the full image resized to 224x224 and log the filename in `results/no_detection.csv`. Report the detection rate.

### 3.3 Landmark branch preprocessing (42-D)

- 21 landmarks x (x, y) = **42-D** for one hand.
- **Normalize**: subtract the wrist landmark (translation invariance), then divide by a hand-size scale such as the wrist-to-middle-finger-MCP distance (scale invariance). Optionally mirror left hands to right so handedness does not matter.
- **Check:** if the dataset contains two-handed signs (many ISL letters use both hands), a single-hand 42-D vector will lose information. In that case, additionally build an **84-D two-hand variant** (zero-filled when only one hand is visible), run it as an ablation, and report both. Record the decision in `docs/ASSUMPTIONS.md`.
- Cache features to `data/landmarks/*.npz`. Rows with no detection are excluded from the landmark branch only, and handled explicitly in fusion (see Section 4.3).

### 3.4 Dataset splitting

The project specification is a **70% train / 30% test** split. Implement exactly that, and carve a **validation set out of the training portion** (for example 15% of train) for early stopping and fusion-weight tuning. **The test set must never be used for tuning.**

Create three split files in `data/splits/`:

| Split | How |
|---|---|
| `standard.csv` | Stratified 70/30 split, seed 42. Use only as the *baseline* protocol. Warn in the README that consecutive frames from the same signer may leak. |
| `signer_independent.csv` | Group split: all images of a signer appear in only one of train, val or test (about 70/30 by signer). Use `GroupShuffleSplit` or `GroupKFold`. |
| `background_independent.csv` | Group split by background or scene: test backgrounds never appear in training. |

If signer or background labels do **not** exist in the dataset, do not fabricate them. Instead: (a) group by session or folder structure if one exists; (b) for background, create a **synthetic held-out test** by replacing backgrounds (using MediaPipe selfie segmentation or a hand mask) with unseen textures and colors; (c) clearly label these as proxy protocols in the README and the report.

---

## 4. Step 2: Model Implementation

### 4.1 CNN branch (two backbones, compared)

- **MobileNetV2** and **EfficientNet-B0**, ImageNet-pretrained.
- Head: `GlobalAveragePooling` then `BatchNorm` then `Dropout(0.3 to 0.5)` then `Dense(35, softmax)`.
- Two-phase training: (1) frozen backbone, train head (about 10 epochs, lr 1e-3); (2) unfreeze the top layers (or whole network at low lr, about 1e-5) and fine-tune.
- Adam, categorical cross-entropy (try label smoothing 0.05 as an ablation), early stopping on validation loss, `ReduceLROnPlateau`.
- **Augmentation (train only):** small rotation, zoom, translation, brightness and contrast jitter, **random background replacement or color jitter**. Do **not** apply horizontal flips unless the sign set is verified as flip-safe (handedness can change the sign).

### 4.2 Landmark DNN

- Input 42-D, `Dense(128, relu)`, `BatchNorm`, `Dropout(0.3)`, `Dense(64, relu)`, `Dense(35, softmax)`.
- Light augmentation on landmarks (small Gaussian noise, small rotation and scaling).
- Early stopping on validation loss.

### 4.3 Hybrid late fusion (main model)

Implement and compare these fusion strategies, all using each branch's **softmax probabilities**:

1. **Weighted average:** `p = w * p_cnn + (1 - w) * p_landmark`; tune `w` on the validation set only.
2. **Stacking:** train a small logistic-regression meta-classifier on concatenated validation probabilities.
3. **Confidence-aware fusion (optional):** when the landmark branch has no detection, fall back to the CNN branch only; when the CNN is low-confidence, weight the landmark branch higher.

Use the best CNN backbone (by validation) as the CNN branch. Also apply **temperature scaling** on the validation set for calibration before fusing.

### 4.4 Baselines (required)

CNN-only and Landmark-only baselines, trained with the same splits, seeds and budget.

### 4.5 Ablations (required)

| Ablation | Purpose |
|---|---|
| MobileNetV2 vs EfficientNet-B0 | Backbone choice |
| With vs without MediaPipe crop | Value of cropping |
| With vs without background augmentation | Background robustness |
| 42-D vs 84-D landmarks (if two-handed signs exist) | Landmark representation |
| Averaging vs stacking vs confidence-aware fusion | Fusion strategy |
| Train from 3 seeds | Report mean and standard deviation |

---

## 5. Step 3: Evaluation Protocol (the main contribution)

Run **every model on all three splits** (standard, signer-independent, background-independent).

**Metrics (all on the held-out test set):**

- Accuracy, Precision, Recall, F1 (**macro and weighted**)
- Cohen's Kappa, Matthews Correlation Coefficient (MCC)
- Per-class precision, recall and F1 table
- Confusion matrix (normalized), plus **top-10 most confused class pairs**
- Model size (MB), parameter count, **inference latency per image (ms) on CPU**
- Statistical check: **McNemar's test** between hybrid and each baseline, and mean plus or minus std over 3 seeds

**Outputs:** `results/metrics_<model>_<split>.json`, `results/summary_table.csv`, confusion-matrix PNGs.

**Generalization gap:** for each model, report `standard accuracy - signer-independent accuracy` and `standard - background-independent`. A smaller gap is evidence of robustness, and this is the key comparison against prior work.

### Best model selection

Select the best model by **signer-independent and background-independent macro-F1** (and report standard-split results too). **Do not label Hybrid Late Fusion as "best" in any text, figure or README unless the measured results show it.** Generate the "Best Model" statement automatically from `results/summary_table.csv`.

---

## 6. Step 4: Explainability

Apply to the hybrid model (CNN branch for Grad-CAM):

- **Grad-CAM:** heatmaps for correct predictions and for misclassified examples. Save a grid of 5 correct and 5 incorrect examples per run in `results/gradcam/`. Check whether attention falls on the hand rather than the background.
- **t-SNE:** 2-D plots of (a) CNN features, (b) landmark features, (c) fused features, colored by class. Use the penultimate-layer features on the test set.
- Add a short written interpretation per plot in `results/explainability_notes.md` (what is separable, which classes overlap, and whether this matches the confusion pairs).

---

## 7. Step 5: Gemini Sign Guidance

Goal: after a prediction, give the user text guidance on how to form the sign correctly.

**Design (to avoid hallucinated guidance):**

1. Create `src/guidance/sign_descriptions.json` with a short, **human-verified** description for each of the 35 classes (finger positions, one or two hands). Mark any entry that still needs verification with `"verified": false`, and leave the final check to me.
2. `gemini_guidance.py` sends Gemini the predicted class, its stored description, the confidence, and (if confidence is low) the **top-2 predicted classes**. The prompt tells Gemini to **use only the supplied description** to give a short, friendly tip, and to explain the difference between the top-2 signs when the model is unsure.
3. Read the key from `GEMINI_API_KEY`. Handle API errors gracefully (return the stored description if Gemini is unavailable).
4. Show guidance only "where applicable" (low confidence or user requests help).

---

## 8. Step 6: Final Model, Outcome and Demo

- **Final model:** the best model according to Section 5 (expected candidate: hybrid CNN plus landmark late fusion with a 35-class softmax). Save weights and `config.yaml` used.
- **Model outcome:** predicted class and confidence for each input hand image.
- **Demo (`app/demo.py`, Streamlit):** upload an image or use the webcam, show the predicted letter and confidence, the Grad-CAM overlay, and the Gemini guidance. For webcam mode, smooth predictions with a majority vote over the last N frames (for example N=5) to reduce flicker.
- **Final output:** recognized sign (one of 35 classes) plus Gemini sign guidance where applicable.

---

## 9. Deliverables Checklist

- [ ] `docs/ASSUMPTIONS.md` (dataset findings: signers, backgrounds, one- or two-handed signs)
- [ ] Three split files, with a leakage check (no overlap of group IDs between train and test for the signer and background splits)
- [ ] Trained: MobileNetV2, EfficientNet-B0, Landmark DNN, Hybrid (all fusion variants)
- [ ] Results table for every model on every split, with mean plus or minus std over 3 seeds
- [ ] Ablation table (Section 4.5)
- [ ] Grad-CAM grid and t-SNE plots with interpretation notes
- [ ] Gemini guidance module with the descriptions JSON
- [ ] Streamlit demo
- [ ] `README.md`: setup, how to run each step, results summary, limitations

## 10. Rules for the Agent

1. Never tune on the test set. Never split by individual frame for the signer and background protocols.
2. Never invent signer or background labels. Use real metadata, or label proxies as proxies.
3. Never claim a "best model" that the metrics do not support.
4. Never commit API keys.
5. Log every run (config, seed, metrics) so the results are reproducible.
6. Report negative results honestly (for example, if fusion does not beat the CNN on some split).

---

## 11. Suggested Positioning for the Report

> Prior work on fingerspelling reports 97% to 100% accuracy, mostly on random splits where the same signer and background appear in both train and test. This project evaluates the same task under **signer-independent and background-independent** protocols, combines **image and landmark evidence by late fusion**, and adds **Grad-CAM, t-SNE and Gemini-based guidance** to make the system interpretable and useful to learners.

(Adjust this wording to match the final measured results.)
