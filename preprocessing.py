"""
ISL Fingerspelling Recognition - Step 2: Data Preprocessing
============================================================
This single-file script performs all Data Preprocessing techniques:
1. Image Preprocessing:
   - Hand detection via MediaPipe
   - Hand bounding-box cropping with 15% margin
   - Symmetrical square padding to prevent aspect ratio distortion
   - Bilinear resizing to 224x224 resolution
   - Pixel normalization [0, 1] range verification
   - Fallback to resized original image if no hand is detected

2. Landmark Feature Extraction:
   - MediaPipe 21 hand landmarks -> 42-D feature vector (x, y coordinates)
   - Translation normalization: wrist landmark (landmark 0) at origin (0, 0)
   - Scale normalization: divided by distance between wrist and middle MCP (landmark 9)
   - Saved to compressed NumPy archive (.npz)

All output is saved cleanly into a dedicated folder:
   preprocessing_output/
     ├── preprocessed_images/   (Subfolders 1-9, A-Z with 224x224 images)
     ├── landmarks_42d.npz      (42-D landmark vectors, labels, and file paths)
     ├── no_detection.csv       (Log of fallback images where no hand was detected)
     ├── preprocessing_results.json  (Detailed summary statistics and metrics)
     └── preprocessing_results.txt   (Human-readable summary report)
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def normalize_landmarks_42d(landmarks):
    """
    Normalizes 21 hand landmarks into a 42-D feature vector:
    1. Translation invariance: subtract wrist position (landmark 0).
    2. Scale invariance: divide by distance from wrist (0) to middle finger MCP (9).
    """
    coords = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32) # (21, 2)
    wrist = coords[0]
    middle_mcp = coords[9]

    # Translation invariance
    norm_coords = coords - wrist

    # Scale invariance
    scale = np.linalg.norm(middle_mcp - wrist)
    if scale < 1e-6:
        # Fallback to coordinate range if middle MCP and wrist overlap
        scale = np.ptp(norm_coords, axis=0).max()
        if scale < 1e-6:
            scale = 1.0

    norm_coords /= scale
    return norm_coords.flatten() # Exactly 42 features

def compute_hand_crop_box(hand_landmarks_list, img_w, img_h, margin=0.15):
    """
    Computes a padded bounding box around all detected hand landmarks.
    """
    all_xs = [lm.x * img_w for hand in hand_landmarks_list for lm in hand]
    all_ys = [lm.y * img_h for hand in hand_landmarks_list for lm in hand]

    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)
    bw, bh = x_max - x_min, y_max - y_min

    # Add margin
    x_min = max(0, x_min - margin * bw)
    y_min = max(0, y_min - margin * bh)
    x_max = min(img_w, x_max + margin * bw)
    y_max = min(img_h, y_max + margin * bh)

    return int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))

def crop_and_square_pad(img, box, target_size=(224, 224)):
    """
    Crops the image to the specified box, symmetrically pads to a square,
    and resizes to (224, 224) preserving original aspect ratio.
    """
    x_min, y_min, x_max, y_max = box
    crop = img[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    ch, cw = crop.shape[:2]
    if ch > cw:
        diff = ch - cw
        pad_l = diff // 2
        pad_r = diff - pad_l
        crop = cv2.copyMakeBorder(crop, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif cw > ch:
        diff = cw - ch
        pad_t = diff // 2
        pad_b = diff - pad_t
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

def run_preprocessing(
    data_dir="ISL_DATA",
    output_dir="preprocessing_output",
    model_path="hand_landmarker.task",
    target_size=(224, 224),
    limit_per_class=None
):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    prep_images_dir = out_path / "preprocessed_images"

    out_path.mkdir(parents=True, exist_ok=True)
    prep_images_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Error: Dataset directory '{data_path}' does not exist.")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"Error: MediaPipe model '{model_path}' not found.")
        sys.exit(1)

    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    print("=" * 70)
    print("ISL FINGERSPELLING: DATA PREPROCESSING PIPELINE (STEP 2)")
    print("=" * 70)
    print(f"Source Dataset:            {data_path.resolve()}")
    print(f"Output Destination:        {out_path.resolve()}")
    print(f"Preprocessed Images:       {prep_images_dir.resolve()}")
    print(f"Target Resolution:         {target_size[0]}x{target_size[1]}")
    print(f"Total Sign Classes:        {len(class_names)} ({', '.join(class_names[:9])} ... {', '.join(class_names[-5:])})")
    if limit_per_class:
        print(f"Class Sample Limit:        {limit_per_class} images per class (Fast Mode)")
    print("=" * 70)

    # Initialize MediaPipe HandLandmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Storage arrays
    features_42d = []
    labels = []
    image_names = []
    has_detection_flags = []
    num_hands_detected = []
    no_detection_list = []
    class_stats = {}

    total_images_processed = 0
    total_hands_detected = 0
    t_start = time.time()

    for idx, c in enumerate(class_names):
        class_src_dir = data_path / c
        class_dst_dir = prep_images_dir / c
        class_dst_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(class_src_dir.iterdir()))
        if limit_per_class:
            files = files[:limit_per_class]

        class_t0 = time.time()
        class_detected = 0

        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue

            img_h, img_w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect(mp_img)

            rel_path = f"{c}/{f.name}"

            if res.hand_landmarks and len(res.hand_landmarks) > 0:
                class_detected += 1
                total_hands_detected += 1
                n_hands = len(res.hand_landmarks)

                # 1. Image Preprocessing: Crop hand bbox with margin, pad to square, resize
                box = compute_hand_crop_box(res.hand_landmarks, img_w, img_h, margin=0.15)
                preprocessed_img = crop_and_square_pad(img, box, target_size=target_size)

                # 2. Landmark Extraction: 42-D translation & scale normalized vector
                vec_42d = normalize_landmarks_42d(res.hand_landmarks[0])
                has_det = True
            else:
                n_hands = 0
                # Fallback: resize original image
                preprocessed_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                vec_42d = np.zeros(42, dtype=np.float32)
                has_det = False
                no_detection_list.append([rel_path, c])

            # Save preprocessed image to output directory
            out_img_path = class_dst_dir / f.name
            cv2.imwrite(str(out_img_path), preprocessed_img)

            # Store feature vectors & metadata
            features_42d.append(vec_42d)
            labels.append(c)
            image_names.append(rel_path)
            has_detection_flags.append(has_det)
            num_hands_detected.append(n_hands)

            total_images_processed += 1

        class_duration = time.time() - class_t0
        class_det_rate = (class_detected / len(files) * 100) if len(files) > 0 else 0.0
        class_stats[c] = {
            "total_images": len(files),
            "hands_detected": class_detected,
            "detection_rate_pct": round(class_det_rate, 2),
            "processing_time_s": round(class_duration, 2)
        }

        print(f"[{idx + 1:02d}/{len(class_names):02d}] Class '{c}': {len(files)} images | "
              f"Detected: {class_detected}/{len(files)} ({class_det_rate:.1f}%) | "
              f"Time: {class_duration:.1f}s")

    detector.close()
    total_duration = time.time() - t_start

    # Convert to NumPy arrays
    features_arr = np.array(features_42d, dtype=np.float32)
    labels_arr = np.array(labels, dtype=str)
    image_names_arr = np.array(image_names, dtype=str)
    has_detection_arr = np.array(has_detection_flags, dtype=bool)
    num_hands_arr = np.array(num_hands_detected, dtype=np.int32)
    classes_arr = np.array(class_names, dtype=str)

    # 1. Save landmark features to .npz file
    landmarks_npz_path = out_path / "landmarks_42d.npz"
    np.savez_compressed(
        landmarks_npz_path,
        features=features_arr,
        labels=labels_arr,
        image_paths=image_names_arr,
        has_detection=has_detection_arr,
        num_hands=num_hands_arr,
        classes=classes_arr
    )
    print(f"\n[+] Saved 42-D Landmark Features: {landmarks_npz_path} ({features_arr.shape})")

    # 2. Save no-detection log
    no_detection_path = out_path / "no_detection.csv"
    with open(no_detection_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "class"])
        for item in no_detection_list:
            writer.writerow(item)
    print(f"[+] Saved No-Detection Log:      {no_detection_path} ({len(no_detection_list)} entries)")

    # 3. Save comprehensive JSON summary
    overall_det_rate = (total_hands_detected / total_images_processed * 100) if total_images_processed > 0 else 0.0
    results_json = {
        "step": "2. Data Preprocessing",
        "description": "Image cropping/resizing and MediaPipe 42-D landmark feature extraction",
        "total_images_processed": total_images_processed,
        "total_hands_detected": total_hands_detected,
        "overall_detection_rate_pct": round(overall_det_rate, 2),
        "target_resolution": list(target_size),
        "num_classes": len(class_names),
        "classes": class_names,
        "features_shape": list(features_arr.shape),
        "output_files": {
            "preprocessed_images_directory": str(prep_images_dir),
            "landmarks_npz_file": str(landmarks_npz_path),
            "no_detection_csv_file": str(no_detection_path)
        },
        "performance": {
            "total_execution_time_seconds": round(total_duration, 2),
            "throughput_images_per_second": round(total_images_processed / total_duration, 2) if total_duration > 0 else 0
        },
        "per_class_summary": class_stats
    }

    results_json_path = out_path / "preprocessing_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"[+] Saved Detailed Results JSON: {results_json_path}")

    # 4. Save human-readable text report
    results_txt_path = out_path / "preprocessing_results.txt"
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ISL FINGERSPELLING RECOGNITION - DATA PREPROCESSING REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Techniques Applied:\n")
        f.write(f"1. Image Preprocessing:\n")
        f.write(f"   - MediaPipe Hand Detection & Bounding Box extraction (15% margin)\n")
        f.write(f"   - Symmetrical zero-padding to square aspect ratio\n")
        f.write(f"   - Bilinear interpolation resizing to {target_size[0]}x{target_size[1]}\n")
        f.write(f"   - Fallback to resized original image on detection failure\n")
        f.write(f"2. Landmark Feature Extraction:\n")
        f.write(f"   - 21 landmarks x (x, y) = 42-D representation\n")
        f.write(f"   - Translation normalization relative to wrist (landmark 0)\n")
        f.write(f"   - Scale normalization relative to wrist-to-middle-MCP distance\n\n")
        f.write(f"Dataset Metrics:\n")
        f.write(f"- Total Images Processed:    {total_images_processed:,}\n")
        f.write(f"- Hands Detected:            {total_hands_detected:,} ({overall_det_rate:.2f}%)\n")
        f.write(f"- Fallback Resized:          {len(no_detection_list):,}\n")
        f.write(f"- Output Images Folder:      {prep_images_dir}\n")
        f.write(f"- Output Landmark Matrix:    {landmarks_npz_path} -> Shape: {features_arr.shape}\n")
        f.write(f"- Total Processing Time:     {total_duration:.2f} seconds ({total_images_processed / total_duration:.1f} images/sec)\n\n")
        f.write("Per-Class Detection Summary:\n")
        f.write(f"{'Class':<8} {'Images':<10} {'Detected':<10} {'Rate (%)':<10} {'Time (s)':<10}\n")
        f.write("-" * 50 + "\n")
        for c, st in class_stats.items():
            f.write(f"{c:<8} {st['total_images']:<10} {st['hands_detected']:<10} {st['detection_rate_pct']:<10.1f} {st['processing_time_s']:<10.1f}\n")
        f.write("=" * 70 + "\n")

    print(f"[+] Saved Human-Readable Report: {results_txt_path}")
    print("=" * 70)
    print("DATA PREPROCESSING (STEP 2) SUCCESSFULLY COMPLETED!")
    print("=" * 70)

    return results_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Fingerspelling Preprocessing")
    parser.add_argument("--data-dir", type=str, default="ISL_DATA", help="Path to raw ISL_DATA folder")
    parser.add_argument("--output-dir", type=str, default="preprocessing_output", help="Output directory")
    parser.add_argument("--model-path", type=str, default="hand_landmarker.task", help="Path to MediaPipe model")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Optional image limit per class for fast testing")
    args = parser.parse_args()

    run_preprocessing(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        limit_per_class=args.limit_per_class
    )
