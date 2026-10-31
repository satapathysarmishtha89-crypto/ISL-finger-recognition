"""
ISL Fingerspelling Recognition - Step 5: Gemini Sign Guidance
===============================================================
Implements Section 7 of the ISL Project Implementation Specification:
1. Grounded Guidance:
   - Uses verified anatomical sign descriptions from sign_descriptions.json
   - Never hallucinates finger positions; prompts Gemini using strictly supplied data
2. Multi-Tier Guidance Logic:
   - High Confidence (>= 0.85): Confirms prediction with a concise posture check
   - Low Confidence (< 0.85) or Ambiguous: Identifies top-2 predictions, explains
     the exact anatomical difference (e.g., U vs V: touching vs spread fingers)
3. Robust Fallback:
   - Reads GEMINI_API_KEY from environment variables (never hardcoded)
   - Gracefully falls back to structured template guidance if API key is absent or network fails

Output Directory:
  step_7_gemini_guidance_output/
    ├── guidance_results.json
    └── guidance_report.txt
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path


def load_sign_descriptions(descriptions_path="sign_descriptions.json"):
    """
    Loads human-verified sign descriptions from JSON file.
    """
    p = Path(descriptions_path)
    if not p.exists():
        raise FileNotFoundError(f"Sign descriptions file not found at: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def format_rule_based_fallback(pred_class, confidence, desc_data, second_class=None, second_desc=None):
    """
    Generates structured, guaranteed text guidance when Gemini API is unavailable.
    """
    info = desc_data.get(pred_class, {})
    sign_desc = info.get("description", "Perform sign with clear finger extension.")
    key_feat = info.get("key_features", "")
    confusion = info.get("common_confusion", "")

    if confidence >= 0.85 or second_class is None:
        guidance = (
            f"Sign '{pred_class}' recognized with high confidence ({confidence*100:.1f}%).\n"
            f"- Correct Form: {sign_desc}\n"
            f"- Pro Tip: Ensure your {key_feat.lower()}."
        )
    else:
        sec_info = desc_data.get(second_class, {})
        sec_desc = sec_info.get("description", "")
        sec_key = sec_info.get("key_features", "")
        guidance = (
            f"The model detected Sign '{pred_class}' ({confidence*100:.1f}%), but it is close to Sign '{second_class}'.\n"
            f"- To form '{pred_class}': {sign_desc}\n"
            f"- Difference from '{second_class}': In '{pred_class}', {key_feat}. In '{second_class}', {sec_key}.\n"
            f"- Action Tip: Check your finger separation and thumb position to remove ambiguity."
        )

    return guidance


def get_gemini_sign_guidance(
    pred_class,
    confidence,
    top2_class=None,
    top2_confidence=None,
    descriptions_path="sign_descriptions.json",
    api_key=None
):
    """
    Generates intelligent sign guidance via Gemini or fallback.
    """
    desc_data = load_sign_descriptions(descriptions_path)
    current_desc = desc_data.get(pred_class, {})

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    # If no key is set, use verified grounded fallback immediately
    if not api_key:
        fallback_text = format_rule_based_fallback(
            pred_class=pred_class,
            confidence=confidence,
            desc_data=desc_data,
            second_class=top2_class,
            second_desc=desc_data.get(top2_class, {}) if top2_class else None
        )
        return {
            "source": "verified_rule_fallback",
            "predicted_class": pred_class,
            "confidence": round(float(confidence), 4),
            "second_class": top2_class,
            "guidance_text": fallback_text
        }

    # Prepare prompt for Gemini
    if confidence < 0.85 and top2_class:
        sec_desc = desc_data.get(top2_class, {})
        system_prompt = (
            "You are an encouraging Indian Sign Language (ISL) tutor. "
            "A student performed a sign, but the computer vision model is uncertain between two similar signs.\n"
            "STRICT INSTRUCTION: Use ONLY the provided anatomical descriptions below. Do NOT invent or hallucinate finger positions.\n\n"
            f"Predicted Sign: '{pred_class}' (Confidence: {confidence*100:.1f}%)\n"
            f"Description for '{pred_class}': {current_desc.get('description', '')}\n"
            f"Key Feature for '{pred_class}': {current_desc.get('key_features', '')}\n\n"
            f"Alternative Sign: '{top2_class}' (Confidence: {top2_confidence*100 if top2_confidence else 0:.1f}%)\n"
            f"Description for '{top2_class}': {sec_desc.get('description', '')}\n"
            f"Key Feature for '{top2_class}': {sec_desc.get('key_features', '')}\n\n"
            "TASK: In 2-3 friendly, concise sentences:\n"
            "1. Explain how to form the intended sign correctly.\n"
            f"2. Clearly highlight the exact physical difference between '{pred_class}' and '{top2_class}'."
        )
    else:
        system_prompt = (
            "You are an encouraging Indian Sign Language (ISL) tutor. "
            "A student performed a sign correctly with high confidence.\n"
            "STRICT INSTRUCTION: Use ONLY the provided anatomical description below.\n\n"
            f"Recognized Sign: '{pred_class}' (Confidence: {confidence*100:.1f}%)\n"
            f"Description: {current_desc.get('description', '')}\n"
            f"Key Feature: {current_desc.get('key_features', '')}\n\n"
            "TASK: Give a short 1-2 sentence encouraging feedback verifying their hand posture and one tip to keep it clear."
        )

    # Attempt calling Gemini API
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(system_prompt)
        guidance_text = response.text.strip()
        source = "gemini-1.5-flash"
    except Exception as e:
        # Graceful fallback on network/quota/auth failure
        fallback_text = format_rule_based_fallback(
            pred_class=pred_class,
            confidence=confidence,
            desc_data=desc_data,
            second_class=top2_class,
            second_desc=desc_data.get(top2_class, {}) if top2_class else None
        )
        return {
            "source": f"verified_fallback (API note: {type(e).__name__})",
            "predicted_class": pred_class,
            "confidence": round(float(confidence), 4),
            "second_class": top2_class,
            "guidance_text": fallback_text
        }

    return {
        "source": source,
        "predicted_class": pred_class,
        "confidence": round(float(confidence), 4),
        "second_class": top2_class,
        "guidance_text": guidance_text
    }


def run_gemini_guidance_pipeline(
    descriptions_path="sign_descriptions.json",
    output_dir="step_7_gemini_guidance_output"
):
    """
    Tests and generates benchmark guidance scenarios for the ISL project.
    """
    t_start = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ISL FINGERSPELLING: STEP 5 - GEMINI SIGN GUIDANCE")
    print("=" * 70)
    print(f"Sign Descriptions:  {descriptions_path}")
    print(f"Output Destination: {out_path.resolve()}")
    print("=" * 70)

    # Define key test scenarios reflecting actual model evaluation conditions
    test_scenarios = [
        {
            "id": "scenario_1_high_confidence",
            "title": "High Confidence Clean Sign (Sign 'A')",
            "pred_class": "A",
            "confidence": 0.998,
            "top2_class": "S",
            "top2_confidence": 0.001
        },
        {
            "id": "scenario_2_ambiguous_loop",
            "title": "Low Confidence Ambiguity: Sign 'D' vs '9'",
            "pred_class": "D",
            "confidence": 0.624,
            "top2_class": "9",
            "top2_confidence": 0.342
        },
        {
            "id": "scenario_3_finger_spread",
            "title": "Low Confidence Ambiguity: Sign 'U' vs 'V'",
            "pred_class": "U",
            "confidence": 0.581,
            "top2_class": "V",
            "top2_confidence": 0.389
        },
        {
            "id": "scenario_4_flat_hand",
            "title": "Confused Palm Sign: Sign 'B' vs 'Q'",
            "pred_class": "B",
            "confidence": 0.642,
            "top2_class": "Q",
            "top2_confidence": 0.315
        },
        {
            "id": "scenario_5_vertical_number",
            "title": "Clean Digit Sign (Sign '1')",
            "pred_class": "1",
            "confidence": 0.995,
            "top2_class": "D",
            "top2_confidence": 0.003
        }
    ]

    results = []
    print("\n[*] Generating Guidance for Benchmark Scenarios...")
    for sc in test_scenarios:
        print(f"\n[*] Evaluating: {sc['title']}")
        res = get_gemini_sign_guidance(
            pred_class=sc["pred_class"],
            confidence=sc["confidence"],
            top2_class=sc["top2_class"],
            top2_confidence=sc["top2_confidence"],
            descriptions_path=descriptions_path
        )
        res["scenario_id"] = sc["id"]
        res["scenario_title"] = sc["title"]
        results.append(res)
        print(f"    - Engine: {res['source']}")
        print(f"    - Guidance Snippet:\n      {res['guidance_text'][:120]}...")

    # 1. Save JSON Outputs
    json_path = out_path / "guidance_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved Guidance Results JSON: {json_path}")

    # 2. Save Human-Readable Text Report
    txt_path = out_path / "guidance_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("ISL FINGERSPELLING: STEP 5 - GEMINI SIGN GUIDANCE REPORT\n")
        f.write("=" * 75 + "\n\n")
        f.write("Specification Implementation Details:\n")
        f.write("- Verified Descriptions:  sign_descriptions.json (35 classes)\n")
        f.write("- Fallback Strategy:      Rule-based verified template when API is absent\n")
        f.write("- Trigger Criteria:       Applied on ambiguous predictions (< 85% conf) or requests\n\n")

        f.write("=" * 75 + "\n")
        f.write("BENCHMARK GUIDANCE SCENARIOS & OUTPUTS\n")
        f.write("=" * 75 + "\n")

        for r in results:
            f.write(f"\n[Scenario]: {r['scenario_title']}\n")
            f.write(f"- Detected Sign:  '{r['predicted_class']}' ({r['confidence']*100:.1f}%)\n")
            if r.get("second_class"):
                f.write(f"- Competing Sign: '{r['second_class']}'\n")
            f.write(f"- Guidance Engine:{r['source']}\n")
            f.write("- Tutor Guidance:\n")
            for line in r["guidance_text"].split("\n"):
                f.write(f"    {line}\n")
            f.write("-" * 75 + "\n")

        f.write("\n===========================================================================\n")
        f.write("STEP 5: GEMINI SIGN GUIDANCE COMPLETED SUCCESSFULLY!\n")
        f.write("===========================================================================\n")

    print(f"[+] Saved Human-Readable Guidance Report: {txt_path}")
    print("=" * 70)
    print("STEP 5: GEMINI SIGN GUIDANCE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Step 5: Gemini Sign Guidance Pipeline")
    parser.add_argument("--descriptions", type=str, default="sign_descriptions.json", help="Path to sign descriptions JSON")
    parser.add_argument("--output-dir", type=str, default="step_7_gemini_guidance_output", help="Output directory")
    args = parser.parse_args()

    run_gemini_guidance_pipeline(
        descriptions_path=args.descriptions,
        output_dir=args.output_dir
    )
