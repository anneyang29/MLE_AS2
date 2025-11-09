# utils/model_monitor.py
# -*- coding: utf-8 -*-
"""
Unified Model Monitoring Module
- Build PSI Baseline
- Calculate Feature PSI
- Calculate Prediction Drift
- Generate Alerts and Notifications
"""

import os
import json
import math
import logging
from typing import List, Dict, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================
NUM_BINS = 10
TOPK_CATS = 50
DROP_DEFAULTS = {"label", "target", "y", "id", "customer_id", "article_id", "user_id", "item_id", "timestamp", "date", "dt"}
EPS = 1e-12

PSI_ALERT_THRESHOLDS = {
    "low": 0.10,           # < 0.10: Green
    "medium": 0.25,        # 0.10-0.25: Yellow
    "high": float("inf")   # >= 0.25: Red
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_spark():
    """Get or create Spark Session"""
    return (SparkSession.builder
            .appName("ModelMonitoring")
            .config("spark.driver.memory", "4g")
            .getOrCreate())

def is_numeric_type(dt):
    """Check if data type is numeric"""
    return dt.typeName() in ["byte", "short", "integer", "long", "float", "double", "decimal"]

def psi_term(p_cur, p_ref, eps=EPS):
    """Calculate single PSI term"""
    p_cur = max(p_cur, eps)
    p_ref = max(p_ref, eps)
    return (p_cur - p_ref) * math.log(p_cur / p_ref)

def grade_psi(psi_value):
    """Grade PSI value"""
    if psi_value < 0.10:
        return "low"
    elif psi_value < 0.25:
        return "medium"
    else:
        return "high"

def get_alert_level(psi_value):
    """Determine alert level based on PSI value"""
    if psi_value < 0.10:
        return "GREEN"
    elif psi_value < 0.25:
        return "YELLOW"
    else:
        return "RED"

# ============================================================================
# PSI BASELINE BUILDING
# ============================================================================

def build_numeric_edges(sdf, col, num_bins=NUM_BINS):
    """Build quantile edges for numeric column"""
    qs = [i / num_bins for i in range(1, num_bins)]
    quantiles = sdf.select(col).approxQuantile(col, qs, 0.001)
    edges = [-float("inf")] + [float(x) for x in quantiles] + [float("inf")]
    return edges

def compute_numeric_pref(sdf, col, edges):
    """Compute reference distribution for numeric column"""
    total = sdf.count() or 1
    conds = []
    for i in range(len(edges) - 1):
        lb, ub = edges[i], edges[i + 1]
        conds.append(F.when((F.col(col) > lb) & (F.col(col) <= ub), F.lit(i)))
    bin_col = F.coalesce(*conds).alias("bin_id")
    dist = (sdf.select(bin_col)
            .groupBy("bin_id").agg(F.count(F.lit(1)).alias("n"))
            .withColumn("p", F.col("n") / F.lit(total))
            .orderBy("bin_id"))
    rows = dist.collect()
    p_map = {int(r["bin_id"]): float(r["p"]) for r in rows if r["bin_id"] is not None}
    p_ref = [p_map.get(i, 0.0) for i in range(len(edges) - 1)]
    return p_ref

def compute_categorical_pref(sdf, col, topk=TOPK_CATS):
    """Compute reference distribution for categorical column"""
    total = sdf.count() or 1
    freq = (sdf.groupBy(col).agg(F.count(F.lit(1)).alias("n")).orderBy(F.desc("n")))
    top = freq.limit(topk).collect()
    top_sum = sum(int(r["n"]) for r in top)
    other_sum = total - top_sum
    p_map = {}
    for r in top:
        key = "__NULL__" if r[col] is None else str(r[col])
        p_map[key] = float(int(r["n"]) / total)
    if other_sum > 0:
        p_map["__OTHER__"] = float(other_sum / total)
    return p_map

def build_psi_baseline(model_name: str, baseline_snapshot: str, x_feature_path: str, 
                       out_baseline_dir: str, numeric_cols: List[str] = None, 
                       categorical_cols: List[str] = None):
    """
    Build PSI Baseline
    
    Args:
        model_name: Model name
        baseline_snapshot: Baseline snapshot date (e.g., "2024-09-01")
        x_feature_path: Path to feature parquet
        out_baseline_dir: Output directory
        numeric_cols: List of numeric columns (optional)
        categorical_cols: List of categorical columns (optional)
    """
    spark = get_spark()
    sdf = spark.read.parquet(x_feature_path)
    
    num_cols_inf = [f.name for f in sdf.schema.fields if is_numeric_type(f.dataType)]
    cat_cols_inf = [f.name for f in sdf.schema.fields if not is_numeric_type(f.dataType)]
    
    if numeric_cols is None:
        numeric_cols = [c for c in num_cols_inf if c not in DROP_DEFAULTS]
    if categorical_cols is None:
        categorical_cols = [c for c in cat_cols_inf if c not in DROP_DEFAULTS]
    
    baseline = {"numeric": {}, "categorical": {}}
    
    logger.info(f"Building baseline for {model_name} - snapshot: {baseline_snapshot}")
    logger.info(f"Numeric columns: {len(numeric_cols)}, Categorical columns: {len(categorical_cols)}")
    
    for c in numeric_cols:
        edges = build_numeric_edges(sdf, c, NUM_BINS)
        p_ref = compute_numeric_pref(sdf.select(c), c, edges)
        baseline["numeric"][c] = {"edges": edges, "p_ref": p_ref}
    
    for c in categorical_cols:
        p_map = compute_categorical_pref(sdf.select(c), c, TOPK_CATS)
        baseline["categorical"][c] = {"p_ref": p_map}
    
    save_dir = os.path.join(out_baseline_dir, model_name, f"snapshot_date={baseline_snapshot}")
    os.makedirs(save_dir, exist_ok=True)
    
    baseline_path = os.path.join(save_dir, "baseline.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Baseline saved to: {baseline_path}")
    return baseline_path

# ============================================================================
# FEATURE PSI CALCULATION
# ============================================================================

def compute_numeric_psi(cur_sdf, feature: str, edges, p_ref_list):
    """Calculate PSI for numeric feature"""
    total = cur_sdf.count() or 1
    conds = []
    for i in range(len(edges) - 1):
        lb, ub = edges[i], edges[i + 1]
        conds.append(F.when((F.col(feature) > lb) & (F.col(feature) <= ub), F.lit(i)))
    bin_col = F.coalesce(*conds).alias("bin_id")
    dist = (cur_sdf.select(bin_col)
            .groupBy("bin_id").agg(F.count(F.lit(1)).alias("n"))
            .withColumn("p_cur", F.col("n") / F.lit(total)))
    p_map = {int(r["bin_id"]): float(r["p_cur"]) for r in dist.collect() if r["bin_id"] is not None}
    p_cur_list = [p_map.get(i, 0.0) for i in range(len(p_ref_list))]
    psi = sum(psi_term(pc, pr) for pc, pr in zip(p_cur_list, p_ref_list))
    return float(psi)

def compute_categorical_psi(cur_sdf, feature: str, p_ref_map: Dict[str, float]):
    """Calculate PSI for categorical feature"""
    total = cur_sdf.count() or 1
    freq = (cur_sdf.groupBy(feature).agg(F.count(F.lit(1)).alias("n")))
    rows = freq.collect()
    p_cur_map, extra = {}, 0.0
    
    for r in rows:
        key = "__NULL__" if r[feature] is None else str(r[feature])
        p = float(int(r["n"]) / total)
        if key in p_ref_map:
            p_cur_map[key] = p_cur_map.get(key, 0.0) + p
        else:
            extra += p
    
    if extra > 0:
        p_cur_map["__OTHER__"] = p_cur_map.get("__OTHER__", 0.0) + extra
    
    psi = 0.0
    for k, p_ref in p_ref_map.items():
        psi += psi_term(p_cur_map.get(k, 0.0), p_ref)
    return float(psi)

def compute_feature_psi(model_name: str, baseline_snapshot: str, current_snapshot: str,
                        current_x_path: str, baseline_dir: str, out_dir: str):
    """
    Calculate feature-level PSI
    """
    spark = get_spark()
    cur = spark.read.parquet(current_x_path)
    
    bl_path = os.path.join(baseline_dir, model_name, f"snapshot_date={baseline_snapshot}", "baseline.json")
    with open(bl_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    
    results = []
    logger.info(f"Calculating feature PSI - baseline: {baseline_snapshot}, current: {current_snapshot}")
    
    for feat, meta in baseline.get("numeric", {}).items():
        if feat not in cur.columns:
            continue
        psi = compute_numeric_psi(cur.select(feat), feat, meta["edges"], meta["p_ref"])
        results.append((model_name, baseline_snapshot, current_snapshot, feat, "numeric", float(psi), grade_psi(float(psi))))
    
    for feat, meta in baseline.get("categorical", {}).items():
        if feat not in cur.columns:
            continue
        psi = compute_categorical_psi(cur.select(feat), feat, meta["p_ref"])
        results.append((model_name, baseline_snapshot, current_snapshot, feat, "categorical", float(psi), grade_psi(float(psi))))
    
    out = spark.createDataFrame(results, schema="""
        model_name STRING,
        baseline_snapshot STRING,
        current_snapshot STRING,
        feature STRING,
        value_type STRING,
        psi DOUBLE,
        psi_grade STRING
    """)
    
    save_path = os.path.join(out_dir, model_name, f"snapshot_date={current_snapshot}")
    (out.coalesce(1).write.mode("overwrite").parquet(save_path))
    
    logger.info(f"Feature PSI saved to: {save_path}")
    return save_path, results

# ============================================================================
# PREDICTION DRIFT DETECTION
# ============================================================================

def calculate_prediction_psi(expected_preds: pd.Series, actual_preds: pd.Series, buckets: int = 10):
    """
    Calculate PSI for prediction values using quantile-based binning
    
    Args:
        expected_preds: Baseline predictions (Series)
        actual_preds: Current predictions (Series)
        buckets: Number of bins
    
    Returns:
        float: PSI value
    """
    _, bins = pd.qcut(expected_preds, q=buckets, retbins=True, duplicates='drop')
    
    expected_binned = pd.cut(expected_preds, bins=bins, include_lowest=True)
    actual_binned = pd.cut(actual_preds, bins=bins, include_lowest=True)
    
    df_expected = pd.DataFrame({'expected': expected_binned.value_counts(normalize=True)})
    df_actual = pd.DataFrame({'actual': actual_binned.value_counts(normalize=True)})
    
    psi_df = df_expected.merge(df_actual, left_index=True, right_index=True, how='outer').fillna(0.00001)
    psi_df['psi'] = (psi_df['actual'] - psi_df['expected']) * np.log(psi_df['actual'] / psi_df['expected'])
    
    psi_value = np.sum(psi_df['psi'])
    return float(psi_value)

def compute_prediction_drift(model_name: str, baseline_pred_path: str, current_pred_path: str,
                             out_dir: str, baseline_pred_col: str = "prediction",
                             current_pred_col: str = "model_predictions"):
    """
    Calculate prediction drift (PSI on predictions)
    
    Args:
        model_name: Model name
        baseline_pred_path: Path to baseline predictions (parquet)
        current_pred_path: Path to current predictions (parquet)
        out_dir: Output directory
        baseline_pred_col: Baseline prediction column name
        current_pred_col: Current prediction column name
    
    Returns:
        dict: Contains PSI value and statistics
    """
    try:
        spark = get_spark()
        
        baseline_df = spark.read.parquet(baseline_pred_path).toPandas()
        current_df = spark.read.parquet(current_pred_path).toPandas()
        
        logger.info(f"Loaded {len(baseline_df)} baseline predictions")
        logger.info(f"Loaded {len(current_df)} current predictions")
        
        # Calculate PSI
        psi_score = calculate_prediction_psi(baseline_df[baseline_pred_col], current_df[current_pred_col])
        
        # Calculate statistics
        baseline_stats = {
            "mean": float(baseline_df[baseline_pred_col].mean()),
            "std": float(baseline_df[baseline_pred_col].std()),
            "min": float(baseline_df[baseline_pred_col].min()),
            "max": float(baseline_df[baseline_pred_col].max()),
            "count": int(len(baseline_df))
        }
        
        current_stats = {
            "mean": float(current_df[current_pred_col].mean()),
            "std": float(current_df[current_pred_col].std()),
            "min": float(current_df[current_pred_col].min()),
            "max": float(current_df[current_pred_col].max()),
            "count": int(len(current_df))
        }
        
        result = {
            "model_name": model_name,
            "psi": psi_score,
            "alert_level": get_alert_level(psi_score),
            "baseline_stats": baseline_stats,
            "current_stats": current_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save result
        os.makedirs(out_dir, exist_ok=True)
        result_path = os.path.join(out_dir, f"{model_name}_prediction_drift.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Prediction drift result saved to: {result_path}")
        logger.info(f"PSI Score: {psi_score:.6f} | Alert: {result['alert_level']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction drift calculation failed: {str(e)}")
        raise

# ============================================================================
# ALERT SYSTEM
# ============================================================================

def generate_alert(model_name: str, psi_value: float, alert_config: Dict = None):
    """
    Generate alert based on PSI value
    
    Args:
        model_name: Model name
        psi_value: PSI value
        alert_config: Alert configuration (optional)
    
    Returns:
        dict: Alert information
    """
    alert_level = get_alert_level(psi_value)
    
    alert_map = {
        "GREEN": {
            "status": "PASS",
            "action": "CONTINUE",
            "message": f"Model is running normally. PSI={psi_value:.6f}",
            "notify": False
        },
        "YELLOW": {
            "status": "WARNING",
            "action": "MONITOR",
            "message": f"Minor drift detected. PSI={psi_value:.6f}, please monitor closely",
            "notify": True
        },
        "RED": {
            "status": "CRITICAL",
            "action": "RETRAIN",
            "message": f"Significant drift detected. PSI={psi_value:.6f}, recommend immediate retraining",
            "notify": True
        }
    }
    
    alert = alert_map.get(alert_level, {})
    alert.update({
        "model_name": model_name,
        "psi": psi_value,
        "alert_level": alert_level,
        "timestamp": datetime.now().isoformat()
    })
    
    return alert

def trigger_notification(alert: Dict, recipients: List[str] = None, use_slack: bool = False):
    """
    Trigger notification (Email/Slack)
    
    Args:
        alert: Alert information
        recipients: Email recipients list
        use_slack: Whether to use Slack
    """
    if not alert.get("notify", False):
        logger.info("No notification needed")
        return
    
    message = f"""
    [{alert['alert_level']} ALERT]
    Model: {alert['model_name']}
    PSI: {alert['psi']:.6f}
    Status: {alert['status']}
    Action: {alert['action']}
    Message: {alert['message']}
    Timestamp: {alert['timestamp']}
    """
    
    logger.warning(message)
    
    if recipients:
        logger.info(f"Sending notification to: {', '.join(recipients)}")
    if use_slack:
        logger.info("Sending Slack notification")

# ============================================================================
# MONITORING PIPELINE
# ============================================================================

def run_monitoring_pipeline(config: Dict):
    """
    Run complete monitoring pipeline
    """
    logger.info("=" * 60)
    logger.info("Starting monitoring pipeline")
    logger.info("=" * 60)
    
    try:
        drift_result = compute_prediction_drift(
            model_name=config["model_name"],
            baseline_pred_path=config["baseline_pred_path"],
            current_pred_path=config["current_pred_path"],
            out_dir=config["out_dir"]
        )
        
        alert = generate_alert(
            model_name=config["model_name"],
            psi_value=drift_result["psi"],
            alert_config=config.get("alert_config")
        )
        
        if alert.get("notify"):
            trigger_notification(
                alert,
                recipients=config.get("recipients"),
                use_slack=config.get("use_slack", False)
            )
        
        logger.info("=" * 60)
        logger.info("Monitoring pipeline completed successfully")
        logger.info("=" * 60)
        
        return drift_result, alert
        
    except Exception as e:
        logger.error(f"Monitoring pipeline failed: {str(e)}")
        raise

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Model Monitoring Module")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Build baseline
    parser_baseline = subparsers.add_parser("build_baseline", help="Build PSI Baseline")
    parser_baseline.add_argument("--model_name", required=True)
    parser_baseline.add_argument("--baseline_snapshot", required=True)
    parser_baseline.add_argument("--x_feature_path", required=True)
    parser_baseline.add_argument("--out_dir", default="datamart/gold/psi_baseline")
    
    # Feature PSI
    parser_feature_psi = subparsers.add_parser("compute_feature_psi", help="Calculate Feature PSI")
    parser_feature_psi.add_argument("--model_name", required=True)
    parser_feature_psi.add_argument("--baseline_snapshot", required=True)
    parser_feature_psi.add_argument("--current_snapshot", required=True)
    parser_feature_psi.add_argument("--current_x_path", required=True)
    parser_feature_psi.add_argument("--baseline_dir", default="datamart/gold/psi_baseline")
    parser_feature_psi.add_argument("--out_dir", default="datamart/gold/psi_monitoring")
    
    # Prediction drift
    parser_pred_drift = subparsers.add_parser("compute_prediction_drift", help="Calculate Prediction Drift")
    parser_pred_drift.add_argument("--model_name", required=True)
    parser_pred_drift.add_argument("--baseline_pred_path", required=True)
    parser_pred_drift.add_argument("--current_pred_path", required=True)
    parser_pred_drift.add_argument("--out_dir", default="datamart/gold/psi_monitoring")
    
    args = parser.parse_args()
    
    if args.command == "build_baseline":
        build_psi_baseline(
            model_name=args.model_name,
            baseline_snapshot=args.baseline_snapshot,
            x_feature_path=args.x_feature_path,
            out_baseline_dir=args.out_dir
        )
    elif args.command == "compute_feature_psi":
        compute_feature_psi(
            model_name=args.model_name,
            baseline_snapshot=args.baseline_snapshot,
            current_snapshot=args.current_snapshot,
            current_x_path=args.current_x_path,
            baseline_dir=args.baseline_dir,
            out_dir=args.out_dir
        )
    elif args.command == "compute_prediction_drift":
        compute_prediction_drift(
            model_name=args.model_name,
            baseline_pred_path=args.baseline_pred_path,
            current_pred_path=args.current_pred_path,
            out_dir=args.out_dir
        )
