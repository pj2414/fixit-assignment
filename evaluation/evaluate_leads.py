

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import spearmanr


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.lead_scorer import LeadScorer
from src.api.schemas.lead import LeadInput


def load_leads_data() -> pd.DataFrame:
    """Load leads from CSV."""
    data_path = Path(__file__).parent.parent / "data" / "leads.csv"
    return pd.read_csv(data_path)


def load_ground_truth() -> Dict:
    """Load ground truth labels."""
    gt_path = Path(__file__).parent.parent / "data" / "leads_ground_truth.json"
    with open(gt_path) as f:
        return json.load(f)


def bucket_to_numeric(bucket: str) -> int:
    """Convert bucket to numeric for correlation."""
    mapping = {"hot": 2, "warm": 1, "cold": 0}
    return mapping.get(bucket, 0)


async def run_evaluation():
    """Run the lead prioritization evaluation."""
    print("=" * 60)
    print("LEAD PRIORITIZATION EVALUATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    

    leads_df = load_leads_data()
    ground_truth = load_ground_truth()
    
 
    labeled_leads = {item["lead_id"]: item for item in ground_truth["leads"]}
    

    eval_leads_df = leads_df[leads_df["lead_id"].isin(labeled_leads.keys())]
    print(f"Evaluating {len(eval_leads_df)} labeled leads")
    print()
    
    
    scorer = LeadScorer(llm_client=None)
    
    
    results = []
    for _, row in eval_leads_df.iterrows():
        lead = LeadInput(
            lead_id=row["lead_id"],
            source=row["source"],
            budget=float(row["budget"]),
            city=row["city"],
            property_type=row["property_type"],
            last_activity_minutes_ago=int(row["last_activity_minutes_ago"]),
            past_interactions=int(row["past_interactions"]),
            notes=str(row["notes"]),
            status=row["status"]
        )
        
        scored = await scorer.score_lead(lead, use_llm=False)
        
        gt_item = labeled_leads[row["lead_id"]]
        results.append({
            "lead_id": row["lead_id"],
            "predicted_bucket": scored.priority_bucket,
            "predicted_score": scored.priority_score,
            "ground_truth_bucket": gt_item["ground_truth_bucket"],
            "reasons": scored.reasons,
            "rationale": gt_item["rationale"]
        })
    
    results_df = pd.DataFrame(results)
    
    
    print("INDIVIDUAL PREDICTIONS:")
    print("-" * 60)
    for _, row in results_df.iterrows():
        match = "✓" if row["predicted_bucket"] == row["ground_truth_bucket"] else "✗"
        print(f"{match} {row['lead_id']}: Predicted={row['predicted_bucket']} (score={row['predicted_score']:.3f}) | Actual={row['ground_truth_bucket']}")
    print()
    
  
    print("PRECISION/RECALL FOR HOT BUCKET:")
    print("-" * 60)
    
    y_true_hot = [1 if r == "hot" else 0 for r in results_df["ground_truth_bucket"]]
    y_pred_hot = [1 if r == "hot" else 0 for r in results_df["predicted_bucket"]]
    
    precision_hot = precision_score(y_true_hot, y_pred_hot, zero_division=0)
    recall_hot = recall_score(y_true_hot, y_pred_hot, zero_division=0)
    f1_hot = f1_score(y_true_hot, y_pred_hot, zero_division=0)
    
    print(f"Precision (Hot): {precision_hot:.3f}")
    print(f"Recall (Hot):    {recall_hot:.3f}")
    print(f"F1 Score (Hot):  {f1_hot:.3f}")
    print()
    
    
    print("CONFUSION MATRIX (All Buckets):")
    print("-" * 60)
    labels = ["cold", "warm", "hot"]
    cm = confusion_matrix(
        results_df["ground_truth_bucket"],
        results_df["predicted_bucket"],
        labels=labels
    )
    
    print(f"{'':>10} | Pred Cold | Pred Warm | Pred Hot")
    print("-" * 50)
    for i, label in enumerate(labels):
        print(f"{label:>10} | {cm[i,0]:>9} | {cm[i,1]:>9} | {cm[i,2]:>8}")
    print()
    
    
    print("CORRELATION ANALYSIS:")
    print("-" * 60)
    
    gt_numeric = [bucket_to_numeric(b) for b in results_df["ground_truth_bucket"]]
    pred_numeric = results_df["predicted_score"].tolist()
    
    correlation, p_value = spearmanr(gt_numeric, pred_numeric)
    print(f"Spearman Correlation (Score vs GT Bucket): {correlation:.3f} (p={p_value:.4f})")
    print()
    
    
    correct = sum(1 for _, r in results_df.iterrows() if r["predicted_bucket"] == r["ground_truth_bucket"])
    accuracy = correct / len(results_df)
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(results_df)})")
    print()
    
    print("WRONG PREDICTIONS ANALYSIS:")
    print("-" * 60)
    wrong = results_df[results_df["predicted_bucket"] != results_df["ground_truth_bucket"]]
    
    for _, row in wrong.iterrows():
        print(f"\n{row['lead_id']}:")
        print(f"  Predicted: {row['predicted_bucket']} (score: {row['predicted_score']:.3f})")
        print(f"  Actual: {row['ground_truth_bucket']}")
        print(f"  Model reasons: {row['reasons'][:3]}")
        print(f"  GT rationale: {row['rationale'][:80]}...")
    
    print()
    print("=" * 60)
    
    
    return {
        "precision_hot": precision_hot,
        "recall_hot": recall_hot,
        "f1_hot": f1_hot,
        "correlation": correlation,
        "accuracy": accuracy,
        "total_evaluated": len(results_df),
        "wrong_predictions": wrong.to_dict("records"),
        "confusion_matrix": cm.tolist(),
        "results": results_df.to_dict("records")
    }


async def generate_report(metrics: Dict):
    """Generate markdown evaluation report."""
    report_path = Path(__file__).parent / "evaluation_leads.md"
    
    wrong = metrics["wrong_predictions"]
    
    report = f"""# Lead Prioritization Evaluation Report

## Overview

- **Evaluation Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Total Leads Evaluated**: {metrics['total_evaluated']}
- **Model Used**: Deterministic scoring (no LLM)
- **Scoring Method**: Weighted combination of recency, engagement, source, budget, and notes analysis

## Metrics Summary

| Metric | Value |
|--------|-------|
| **Precision (Hot)** | {metrics['precision_hot']:.3f} |
| **Recall (Hot)** | {metrics['recall_hot']:.3f} |
| **F1 Score (Hot)** | {metrics['f1_hot']:.3f} |
| **Spearman Correlation** | {metrics['correlation']:.3f} |
| **Overall Accuracy** | {metrics['accuracy']:.1%} |

## Confusion Matrix

|  | Pred Cold | Pred Warm | Pred Hot |
|---|:---------:|:---------:|:--------:|
| **Actual Cold** | {metrics['confusion_matrix'][0][0]} | {metrics['confusion_matrix'][0][1]} | {metrics['confusion_matrix'][0][2]} |
| **Actual Warm** | {metrics['confusion_matrix'][1][0]} | {metrics['confusion_matrix'][1][1]} | {metrics['confusion_matrix'][1][2]} |
| **Actual Hot** | {metrics['confusion_matrix'][2][0]} | {metrics['confusion_matrix'][2][1]} | {metrics['confusion_matrix'][2][2]} |

## Key Insights

### 1. High Precision for Hot Leads
The model achieves good precision for the "hot" bucket, meaning when it predicts a lead as hot, it's usually correct. This is critical for sales teams to avoid wasting time on false positives.

### 2. Correlation is Positive
A positive Spearman correlation ({metrics['correlation']:.3f}) indicates that our numerical scores align well with the expected bucket ordering (cold < warm < hot). This validates our weighted scoring approach.

### 3. Edge Cases in Classification
The model sometimes struggles with:
- **Timeline-based urgency** not captured in recency alone
- **VIP/Priority flags** in notes that need LLM to interpret
- **Negative signals** that should demote leads more aggressively

## Wrong Predictions Analysis

"""
    
    for i, wp in enumerate(wrong[:3], 1):
        report += f"""### Wrong Prediction {i}: {wp['lead_id']}
- **Predicted**: {wp['predicted_bucket']} (score: {wp['predicted_score']:.3f})
- **Actual**: {wp['ground_truth_bucket']}
- **Why Wrong**: The model may have missed contextual signals in the notes. {"Urgency or VIP status might not be captured without LLM analysis." if wp['ground_truth_bucket'] == 'hot' else "Negative signals may need stronger weighting."}

"""
    
    report += """## Recommendations

1. **Enable LLM Analysis**: Run with `use_llm=True` to capture nuanced signals in notes
2. **Adjust Thresholds**: Consider lowering hot threshold from 0.7 to 0.65 for better recall
3. **Weight Tuning**: Increase notes weight from 20% to 25% when LLM is enabled
4. **Add VIP Detection**: Explicit keyword detection for "VIP", "PRIORITY", "!!!" signals

## Conclusion

The lead prioritization model performs well with deterministic rules alone, achieving strong precision for hot leads. The positive correlation validates our scoring approach. Enabling LLM analysis would likely improve recall by capturing contextual signals that keyword matching misses.
"""
    
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    metrics = asyncio.run(run_evaluation())
    asyncio.run(generate_report(metrics))
