

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schemas.call import CallEvalRequest, CallEvalResponse, CallLabels, ModelMetadata


def load_calls_data() -> List[Dict]:
    """Load calls from JSON."""
    data_path = Path(__file__).parent.parent / "data" / "calls.json"
    with open(data_path) as f:
        return json.load(f)


def load_ground_truth() -> Dict:
    """Load ground truth labels."""
    gt_path = Path(__file__).parent.parent / "data" / "calls_ground_truth.json"
    with open(gt_path) as f:
        return json.load(f)


def simulate_call_evaluation(transcript: str) -> Dict:
    """
    Simulate call evaluation without actual LLM.
    Uses heuristic-based scoring for offline evaluation.
    """
    transcript_lower = transcript.lower()
    
    
    rapport_score = 0.5
    if any(word in transcript_lower for word in ["good morning", "good afternoon", "hello", "hi"]):
        rapport_score += 0.1
    if any(word in transcript_lower for word in ["sir", "ma'am", "mr.", "mrs."]):
        rapport_score += 0.1
    if any(word in transcript_lower for word in ["understand", "appreciate", "thank you"]):
        rapport_score += 0.1
    if any(word in transcript_lower for word in ["personally", "definitely", "absolutely"]):
        rapport_score += 0.1
    
    
    need_score = 0.5
    if "?" in transcript:
        need_score += min(0.3, transcript.count("?") * 0.05)
    if any(word in transcript_lower for word in ["requirement", "need", "looking for", "budget"]):
        need_score += 0.15
    if any(word in transcript_lower for word in ["prefer", "priority", "important"]):
        need_score += 0.1
    
    
    closing_score = 0.3
    if any(word in transcript_lower for word in ["visit", "schedule", "book", "when can"]):
        closing_score += 0.2
    if any(word in transcript_lower for word in ["saturday", "sunday", "tomorrow", "weekend"]):
        closing_score += 0.15
    if any(word in transcript_lower for word in ["send", "whatsapp", "email"]):
        closing_score += 0.1
    if any(word in transcript_lower for word in ["looking forward", "see you"]):
        closing_score += 0.15
    
   
    compliance_risk = 0.1
    if any(word in transcript_lower for word in ["guarantee", "promise", "definitely will"]):
        compliance_risk += 0.2
    if any(word in transcript_lower for word in ["pressure", "today only", "last chance"]):
        compliance_risk += 0.3
    if "bye" in transcript_lower and len(transcript) < 200:
        compliance_risk += 0.1  
    
    
    rapport_score = min(1.0, max(0.0, rapport_score))
    need_score = min(1.0, max(0.0, need_score))
    closing_score = min(1.0, max(0.0, closing_score))
    compliance_risk = min(1.0, max(0.0, compliance_risk))
    
    
    quality_score = (
        rapport_score * 0.25 +
        need_score * 0.30 +
        closing_score * 0.30 +
        (1 - compliance_risk) * 0.15
    )
    
    return {
        "labels": {
            "rapport_building": round(rapport_score, 3),
            "need_discovery": round(need_score, 3),
            "closing_attempt": round(closing_score, 3),
            "compliance_risk": round(compliance_risk, 3)
        },
        "quality_score": round(quality_score, 3),
        "is_good_call": quality_score >= 0.6
    }


def find_optimal_threshold(results: List[Dict]) -> float:
    """Find threshold that maximizes F1 score."""
    best_f1 = 0
    best_threshold = 0.5
    
    y_true = [1 if r["ground_truth"] == "good" else 0 for r in results]
    
    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        y_pred = [1 if r["quality_score"] >= threshold else 0 for r in results]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold


def run_evaluation():
    """Run the call quality evaluation."""
    print("=" * 60)
    print("CALL QUALITY EVALUATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
   
    calls = load_calls_data()
    ground_truth = load_ground_truth()
    
   
    labeled_calls = {item["call_id"]: item for item in ground_truth["calls"]}
    
    
    eval_calls = [c for c in calls if c["call_id"] in labeled_calls]
    print(f"Evaluating {len(eval_calls)} labeled calls")
    print()
    
   
    results = []
    for call in eval_calls:
        eval_result = simulate_call_evaluation(call["transcript"])
        gt_item = labeled_calls[call["call_id"]]
        
        results.append({
            "call_id": call["call_id"],
            "quality_score": eval_result["quality_score"],
            "predicted_good": eval_result["is_good_call"],
            "ground_truth": gt_item["ground_truth"],
            "labels": eval_result["labels"],
            "rationale": gt_item["rationale"],
            "was_deal_closed": call.get("was_deal_closed", False)
        })
    
    
    optimal_threshold = find_optimal_threshold(results)
    print(f"Optimal Threshold: {optimal_threshold}")
    default_threshold = 0.6
    print(f"Using Default Threshold: {default_threshold}")
    print()
    
   
    print("INDIVIDUAL PREDICTIONS:")
    print("-" * 60)
    for r in results:
        predicted = "good" if r["predicted_good"] else "bad"
        match = "✓" if predicted == r["ground_truth"] else "✗"
        print(f"{match} {r['call_id']}: Predicted={predicted} (score={r['quality_score']:.3f}) | Actual={r['ground_truth']}")
    print()
    
   
    y_true = [1 if r["ground_truth"] == "good" else 0 for r in results]
    y_pred = [1 if r["predicted_good"] else 0 for r in results]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("METRICS:")
    print("-" * 60)
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print()
    
   
    cm = confusion_matrix(y_true, y_pred)
    print("CONFUSION MATRIX:")
    print("-" * 60)
    print(f"{'':>10} | Pred Bad | Pred Good")
    print("-" * 35)
    print(f"{'Actual Bad':>10} | {cm[0,0]:>8} | {cm[0,1]:>9}")
    print(f"{'Actual Good':>10} | {cm[1,0]:>8} | {cm[1,1]:>9}")
    print()
    
    
    print("WRONG PREDICTIONS ANALYSIS:")
    print("-" * 60)
    
    wrong_predictions = []
    for r in results:
        predicted = "good" if r["predicted_good"] else "bad"
        if predicted != r["ground_truth"]:
            wrong_predictions.append(r)
            print(f"\n{r['call_id']}:")
            print(f"  Predicted: {predicted} (score: {r['quality_score']:.3f})")
            print(f"  Actual: {r['ground_truth']}")
            print(f"  Labels: R={r['labels']['rapport_building']:.2f}, N={r['labels']['need_discovery']:.2f}, C={r['labels']['closing_attempt']:.2f}, Risk={r['labels']['compliance_risk']:.2f}")
            print(f"  GT Rationale: {r['rationale'][:100]}...")
    
    print()
    print("=" * 60)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": default_threshold,
        "optimal_threshold": optimal_threshold,
        "confusion_matrix": cm.tolist(),
        "wrong_predictions": wrong_predictions,
        "results": results,
        "total_evaluated": len(results)
    }


def generate_report(metrics: Dict):
    """Generate markdown evaluation report."""
    report_path = Path(__file__).parent / "evaluation_calls.md"
    
    wrong = metrics["wrong_predictions"]
    
    report = f"""# Call Quality Evaluation Report

## Overview

- **Evaluation Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Total Calls Evaluated**: {metrics['total_evaluated']}
- **Model Used**: Heuristic-based scoring (simulated LLM)
- **Threshold Used**: {metrics['threshold']} (optimal found: {metrics['optimal_threshold']})

## Metrics Summary

| Metric | Value |
|--------|-------|
| **Accuracy** | {metrics['accuracy']:.3f} |
| **Precision** | {metrics['precision']:.3f} |
| **Recall** | {metrics['recall']:.3f} |
| **F1 Score** | {metrics['f1']:.3f} |

## Confusion Matrix

|  | Predicted Bad | Predicted Good |
|---|:-------------:|:--------------:|
| **Actual Bad** | {metrics['confusion_matrix'][0][0]} | {metrics['confusion_matrix'][0][1]} |
| **Actual Good** | {metrics['confusion_matrix'][1][0]} | {metrics['confusion_matrix'][1][1]} |

## Wrong Predictions Analysis

"""
    
    for i, wp in enumerate(wrong[:2], 1):
        predicted = "good" if wp["predicted_good"] else "bad"
        report += f"""### Wrong Prediction {i}: {wp['call_id']}

- **Predicted**: {predicted} (score: {wp['quality_score']:.3f})
- **Actual**: {wp['ground_truth']}
- **Dimension Scores**:
  - Rapport Building: {wp['labels']['rapport_building']:.2f}
  - Need Discovery: {wp['labels']['need_discovery']:.2f}
  - Closing Attempt: {wp['labels']['closing_attempt']:.2f}
  - Compliance Risk: {wp['labels']['compliance_risk']:.2f}
- **Ground Truth Rationale**: {wp['rationale']}

**Why the model got it wrong**:
"""
        if predicted == "good" and wp["ground_truth"] == "bad":
            report += """The model failed to detect subtle negative signals like dismissiveness, lack of empathy, or inappropriate handling. The heuristic approach focuses on keyword presence but misses tone and context.

"""
        else:
            report += """The model may have undervalued positive signals in the transcript. The agent's helpful behavior wasn't fully captured by keyword matching, which requires semantic understanding.

"""
    
    report += """## Key Insights

### 1. Keyword-Based Limitations
The heuristic scoring captures explicit signals (greetings, questions, scheduling) but misses:
- **Tone and empathy** - Can't detect condescending vs. warm tone
- **Customer satisfaction** - Customer's implied happiness/frustration
- **Appropriateness** - Whether responses were suitable for the situation

### 2. Rapport Building Detection
The model performs reasonably on rapport building detection, as formal greetings and polite language are keyword-detectable.

### 3. Compliance Risk Underdetection
Subtle pressure tactics and inappropriate guarantees are hard to detect without LLM understanding. The model may miss phrases like "you'll regret not buying now" or pushy follow-ups.

## Recommendations

1. **Use LLM for Production**: Replace heuristics with actual LLM analysis for better semantic understanding
2. **Add Sentiment Analysis**: Incorporate customer sentiment in the quality score
3. **Train on Edge Cases**: Use wrong predictions to improve prompt engineering
4. **Consider Call Length**: Very short calls often indicate poor quality handling

## Conclusion

The call quality evaluation model achieves reasonable accuracy using deterministic heuristics. However, production deployment should use LLM analysis for better detection of:
- Tone and empathy
- Contextual appropriateness
- Subtle compliance violations
- Customer satisfaction signals

The wrong predictions highlight the limitations of keyword matching and the value of semantic understanding.
"""
    
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    metrics = run_evaluation()
    generate_report(metrics)
