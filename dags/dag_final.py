# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import json
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pathlib import Path
import re

from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.exceptions import AirflowSkipException
from airflow.utils.trigger_rule import TriggerRule

from pyspark.sql import SparkSession

def safe_get_variable(key: str, default=None):
    try:
        from airflow.models import Variable
        return Variable.get(key)
    except Exception as e:
        print(f"[WARN] Variable.get('{key}') failed: {e}. Use default={default}")
        return default

def get_float_var(key: str, default_str: str) -> float:
    v = safe_get_variable(key, default_str)
    try:
        return float(v)
    except Exception:
        return float(default_str)

# -----------------------------------------------------------------------------
# Project imports path
# -----------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processing_bronze_table import (
    process_bronze_loan_table,
    process_bronze_clickstream_table,
    process_bronze_attributes_table,
    process_bronze_financials_table,
)
from utils.data_processing_silver_table import (
    process_silver_loan_table,
    process_silver_clickstream_table,
    process_silver_attributes_table,
    process_silver_financials_table,
)
from utils.data_processing_gold_table import (
    process_labels_gold_table,
    process_fts_gold_engag_table,
    process_fts_gold_cust_risk_table,
)

# -----------------------------------------------------------------------------
# Paths / Variables
# -----------------------------------------------------------------------------
MODEL_BANK = "/opt/airflow/model_bank"
ACTIVE_MODEL_VAR = "active_model_uri"
ACTIVE_BASELINE_VAR = "active_baseline_uri"

PSI_OUTDIR = "/opt/airflow/datamart/gold/psi_monitoring"
PSI_HISTORY = os.path.join(MODEL_BANK, "metrics", "psi_history.csv")

# Thresholds (configurable via Airflow Variables)
PSI_THRESHOLD = float(Variable.get("PSI_THRESHOLD", default_var="0.25"))
PSI_KEYCOL_THRESHOLD = float(Variable.get("PSI_KEYCOL_THRESHOLD", default_var="0.20"))
PSI_KEYCOL_MIN_COUNT = int(Variable.get("PSI_KEYCOL_MIN_COUNT", default_var="2"))
RETRAIN_COOLDOWN_DAYS = int(Variable.get("RETRAIN_COOLDOWN_DAYS", default_var="14"))

# -----------------------------------------------------------------------------
# Spark
# -----------------------------------------------------------------------------
def get_spark_session():
    return (
        SparkSession.builder
        .appName("DataPipeline")
        .master("local[*]")
        .config("spark.sql.execution.arrow.enabled", "false")
        .config("spark.sql.execution.pythonUDF.arrow.enabled", "false")
        .config("spark.python.worker.ignoreVersion", "true")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )

# -----------------------------------------------------------------------------
# Bronze / Silver / Gold wrappers
# -----------------------------------------------------------------------------
def bronze_loan_wrapper(**kwargs):
    spark = get_spark_session()
    process_bronze_loan_table(kwargs['ds'], '/opt/airflow/datamart/bronze/lms/', spark)
    spark.stop()

def bronze_clickstream_wrapper(**kwargs):
    spark = get_spark_session()
    process_bronze_clickstream_table(kwargs['ds'], '/opt/airflow/datamart/bronze/clks/', spark)
    spark.stop()

def bronze_attributes_wrapper(**kwargs):
    spark = get_spark_session()
    process_bronze_attributes_table(kwargs['ds'], '/opt/airflow/datamart/bronze/attr/', spark)
    spark.stop()

def bronze_financials_wrapper(**kwargs):
    spark = get_spark_session()
    process_bronze_financials_table(kwargs['ds'], '/opt/airflow/datamart/bronze/fin/', spark)
    spark.stop()

def silver_loan_wrapper(**kwargs):
    spark = get_spark_session()
    process_silver_loan_table(kwargs['ds'], '/opt/airflow/datamart/bronze/lms/', '/opt/airflow/datamart/silver/lms/', spark)
    spark.stop()

def silver_clickstream_wrapper(**kwargs):
    spark = get_spark_session()
    process_silver_clickstream_table(kwargs['ds'], '/opt/airflow/datamart/bronze/clks/', '/opt/airflow/datamart/silver/clks/', spark)
    spark.stop()

def silver_attributes_wrapper(**kwargs):
    spark = get_spark_session()
    process_silver_attributes_table(kwargs['ds'], '/opt/airflow/datamart/bronze/attr/', '/opt/airflow/datamart/silver/attr/', spark)
    spark.stop()

def silver_financials_wrapper(**kwargs):
    spark = get_spark_session()
    process_silver_financials_table(kwargs['ds'], '/opt/airflow/datamart/bronze/fin/', '/opt/airflow/datamart/silver/fin/', spark)
    spark.stop()

def gold_labels_wrapper(**kwargs):
    spark = get_spark_session()
    process_labels_gold_table(kwargs['ds'], '/opt/airflow/datamart/silver/lms/', '/opt/airflow/datamart/gold/label_store/', spark, dpd=30, mob=3)
    spark.stop()

def gold_engagement_wrapper(**kwargs):
    spark = get_spark_session()
    process_fts_gold_engag_table(kwargs['ds'], '/opt/airflow/datamart/silver/clks/', '/opt/airflow/datamart/gold/ft_store/', spark)
    spark.stop()

def gold_cust_risk_wrapper(**kwargs):
    spark = get_spark_session()
    process_fts_gold_cust_risk_table(kwargs['ds'], '/opt/airflow/datamart/silver/fin/', '/opt/airflow/datamart/gold/ft_store/', spark)
    spark.stop()

# -----------------------------------------------------------------------------
# Stage 4: Training + Activation
# -----------------------------------------------------------------------------
def branch_initial_train_func():
    # If you already have an active model configured, skip initial training
    active_model = Variable.get(ACTIVE_MODEL_VAR, default_var=None)
    if active_model and os.path.exists(active_model):
        print("[INIT] Active model variable exists. Skip initial training.")
        return "skip_initial_train"
    print("[INIT] No active model variable. Proceed to initial training.")
    return "train_model"

def train_model_wrapper(**kwargs):
    """
    Train model for snapshot_date (ds).
    - Validates required paths
    - Checks historical data availability
    - Calls utils.model_train.main(ds) which should return model path and baseline path
    - Pushes XCom: model_uri, baseline_uri
    """
    print("entering train_model_wrapper")
    from dateutil.relativedelta import relativedelta
    print("finish importing dateutil")
    ds = kwargs['ds']
    snapshot_date = datetime.strptime(ds, "%Y-%m-%d")
    print("finish ds snapshot_date")
    required_paths = [
        "/opt/airflow/datamart/gold/label_store",
        "/opt/airflow/datamart/silver/attr",
        "/opt/airflow/datamart/silver/fin",
        "/opt/airflow/model_bank",
    ]
    print("finish required_paths")
    missing = [p for p in required_paths if not os.path.exists(p)]
    print("finish missing")
    if missing:
        # No model will be produced; push state for downstream
        kwargs["ti"].xcom_push(key="model_uri", value=None)
        kwargs["ti"].xcom_push(key="baseline_uri", value=None)
        raise FileNotFoundError(f"Missing required paths: {missing}")
    print("finish if missing")
    # Data sufficiency check: 12m train+test + 2m OOT starting from data_start
    oot_start = snapshot_date - relativedelta(months=2)
    train_test_start = oot_start - relativedelta(months=12)
    data_start = datetime(2023, 1, 1)
    has_enough_data = train_test_start >= data_start
    print("finish data suff check")
    print("============================================================================")
    print(f"TRAINING FEASIBILITY CHECK for {ds}")
    print("============================================================================")
    print(f"Training period start: {train_test_start.strftime('%Y-%m-%d')}")
    print(f"Data available from:   {data_start.strftime('%Y-%m-%d')}")

    if not has_enough_data:
        print("SKIP TRAINING: Not enough historical data.")
        kwargs["ti"].xcom_push(key="model_uri", value=None)
        kwargs["ti"].xcom_push(key="baseline_uri", value=None)
        kwargs["ti"].xcom_push(key="train_status", value="skipped_insufficient_data")
        return {"status": "skipped", "reason": "insufficient_data", "will_start": "2024-01-01"}

    print("TRAINING: Sufficient data available")
    try:
        if "/opt/airflow" not in sys.path:
            sys.path.insert(0, "/opt/airflow")
        os.chdir("/opt/airflow")

        # Your training entrypoint: ideally returns (model_uri, baseline_uri)
        # Update your utils/model_train.main to return these paths.
        from utils.model_train import main as train_main
        result = train_main(ds)

        model_uri = None
        baseline_uri = None

        if isinstance(result, (list, tuple)) and len(result) >= 2:
            model_uri, baseline_uri = result[0], result[1]
        elif isinstance(result, str):
            model_uri = result

        # Fallbacks if train_main didn't return both
        if not model_uri:
            pkl_list = sorted(glob.glob(os.path.join(MODEL_BANK, "credit_model_*.pkl")))
            if not pkl_list:
                kwargs["ti"].xcom_push(key="model_uri", value=None)
                kwargs["ti"].xcom_push(key="baseline_uri", value=None)
                raise FileNotFoundError("No model artifact produced in model_bank/")
            model_uri = pkl_list[-1]

        if not baseline_uri:
            # Heuristic: try to find a baseline parquet in the same bank with similar prefix
            # e.g., credit_model_YYYY_MM_DD.pkl -> credit_model_YYYY_MM_DD_psi_ref_preds.parquet
            base = os.path.splitext(os.path.basename(model_uri))[0]
            cand = os.path.join(MODEL_BANK, f"{base}_psi_ref_preds.parquet")
            if os.path.exists(cand):
                baseline_uri = cand
            else:
                # If still not found, try latest parquet under model_bank
                parquet_list = sorted(glob.glob(os.path.join(MODEL_BANK, "*psi*_preds*.parquet")))
                baseline_uri = parquet_list[-1] if parquet_list else None

        kwargs["ti"].xcom_push(key="model_uri", value=model_uri)
        kwargs["ti"].xcom_push(key="baseline_uri", value=baseline_uri)
        print(f"TRAINING COMPLETED for {ds}. model_uri={model_uri}, baseline_uri={baseline_uri}")
        return {"status": "completed", "snapshot": ds, "model_uri": model_uri, "baseline_uri": baseline_uri}
    except Exception as e:
        print(f"MODEL TRAINING FAILED: {str(e)}")
        raise

def activate_and_set_active_uri(source_task_id: str, **kwargs):
    ti = kwargs["ti"]
    model_uri = ti.xcom_pull(task_ids=source_task_id, key="model_uri")
    baseline_uri = ti.xcom_pull(task_ids=source_task_id, key="baseline_uri")

    if not model_uri:
        raise AirflowSkipException(f"No new model to activate from task '{source_task_id}'. Skipping activation.")
    if not os.path.exists(model_uri):
        raise AirflowSkipException(f"Model path does not exist: {model_uri}. Skipping activation.")

    Variable.set(ACTIVE_MODEL_VAR, model_uri)
    print(f"[ACTIVATE] Variable[{ACTIVE_MODEL_VAR}] = {model_uri}")

    if baseline_uri and os.path.exists(baseline_uri):
        Variable.set(ACTIVE_BASELINE_VAR, baseline_uri)
        print(f"[ACTIVATE] Variable[{ACTIVE_BASELINE_VAR}] = {baseline_uri}")
    else:
        Variable.set(ACTIVE_BASELINE_VAR, "")

    # >>> 關鍵：推 XCom，讓下游 inference 可直接取用 <<<
    ti.xcom_push(key="model_uri", value=model_uri)
    ti.xcom_push(key="baseline_uri", value=baseline_uri if baseline_uri else None)

# -----------------------------------------------------------------------------
# Stage 5: Inference (uses Variables)
# -----------------------------------------------------------------------------
def _latest_model_from_bank(bank_dir: str):
    cands = sorted(glob.glob(os.path.join(bank_dir, "credit_model_*.pkl")))
    return cands[-1] if cands else None

def _guess_baseline_for_model(model_path: str):
    if not model_path:
        return None
    base = os.path.splitext(os.path.basename(model_path))[0]
    cand = os.path.join(MODEL_BANK, f"{base}_psi_ref_preds.parquet")
    if os.path.exists(cand):
        return cand
    plist = sorted(glob.glob(os.path.join(MODEL_BANK, "*psi*_preds*.parquet")))
    return plist[-1] if plist else None

def _safe_var(key, default=None):
    try:
        v = Variable.get(key)
        if isinstance(v, str):
            v = v.strip()
        return v or default
    except Exception:
        return default

def check_model_exists_func(**kwargs):
    ti = kwargs["ti"]

    model_uri    = _safe_var(ACTIVE_MODEL_VAR, None)
    baseline_uri = _safe_var(ACTIVE_BASELINE_VAR, None)

    print(f"[CHECK] Variable model={model_uri}, baseline={baseline_uri}")
    print(f"[CHECK] exists(model)={os.path.exists(model_uri) if model_uri else None}, "
          f"exists(base)={os.path.exists(baseline_uri) if baseline_uri else None}")

    model_ok = bool(model_uri) and os.path.exists(model_uri)
    base_ok  = bool(baseline_uri) and os.path.exists(baseline_uri)

    if not model_ok:
        fb = _latest_model_from_bank(MODEL_BANK)
        print(f"[CHECK] fallback latest model from bank: {fb}")
        if fb:
            model_uri = fb
            model_ok = True

    if model_ok and not base_ok:
        fb_base = _guess_baseline_for_model(model_uri)
        print(f"[CHECK] fallback baseline guess: {fb_base}")
        if fb_base:
            baseline_uri = fb_base
            base_ok = True

    # 推 XCom（讓 inference 可用）
    ti.xcom_push(key="model_uri", value=model_uri if model_ok else None)
    ti.xcom_push(key="baseline_uri", value=baseline_uri if base_ok else None)

    if model_ok:
        print(f"[CHECK] Use model={model_uri}, baseline={baseline_uri if base_ok else '(none)'}")
        return "inference_scoring"

    print("[CHECK] No valid model; skip inference.")
    return "skip_inference"

def resolve_model_for_inference(**kwargs):
    ti = kwargs["ti"]

    # 先試「剛剛訓練完」的 XCom（你現在未啟用 activate_model_initial，就直接讀 train_model）
    model_uri = ti.xcom_pull(task_ids="train_model", key="model_uri")
    baseline_uri = ti.xcom_pull(task_ids="train_model", key="baseline_uri")

    # 若上面為空，再試「檢查現有變數」
    if not model_uri:
        model_uri = _safe_var(ACTIVE_MODEL_VAR, None)
        baseline_uri = _safe_var(ACTIVE_BASELINE_VAR, None)

    # 若還是沒有，嘗試 model_bank 裡最新的
    if not model_uri:
        fb = _latest_model_from_bank(MODEL_BANK)
        if fb:
            model_uri = fb
    if model_uri and not baseline_uri:
        baseline_uri = _guess_baseline_for_model(model_uri)

    print(f"[RESOLVE] model_uri={model_uri}, baseline_uri={baseline_uri}")
    ti.xcom_push(key="model_uri", value=model_uri if model_uri and os.path.exists(model_uri) else None)
    ti.xcom_push(key="baseline_uri", value=baseline_uri if baseline_uri and os.path.exists(baseline_uri) else None)
    return {"status": "ok" if (model_uri and os.path.exists(model_uri)) else "no_model"}


# -----------------------------------------------------------------------------
# Stage 6: PSI + Retrain Branch + Cooldown
# -----------------------------------------------------------------------------
def _psi_one_col(expected, actual, bins=10):
    eps = 1e-9
    q = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, q))
    eh, _ = np.histogram(expected, bins=cuts)
    ah, _ = np.histogram(actual, bins=cuts)
    er = (eh / (eh.sum() + eps)) + eps
    ar = (ah / (ah.sum() + eps)) + eps
    return float(np.sum((ar - er) * np.log(ar / er)))

def _find_prediction_files(month: str):
    base = Path("/opt/airflow/datamart/gold/model_predictions")
    ym_us = month.replace("-", "_")  # "2024-04" -> "2024_04"

    candidates = []

    # 1) 目錄分區：date=YYYY-MM
    p = base / f"date={month}"
    if p.exists():
        candidates += list(p.glob("*.parquet"))

    # 2) 目錄分區：snapshot_date=YYYY-MM-01
    p2 = base / f"snapshot_date={month}-01"
    if p2.exists():
        candidates += list(p2.glob("*.parquet"))

    # 3) 根目錄單檔命名：*preds_YYYY_MM_01.parquet 或 *preds_YYYY_MM.parquet
    candidates += list(base.glob(f"*preds_{ym_us}_*.parquet"))
    candidates += list(base.glob(f"*preds_{ym_us}.parquet"))

    # 去重、轉字串
    seen, result = set(), []
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result

def compute_psi_func(ds, **_):
    month = ds[:7]

    baseline_path = Variable.get(ACTIVE_BASELINE_VAR, default_var=None)
    if not baseline_path or not os.path.exists(baseline_path):
        os.makedirs(PSI_OUTDIR, exist_ok=True)
        payload = {"month": month, "status": "SKIPPED_NO_BASELINE"}
        with open(os.path.join(PSI_OUTDIR, f"psi_{month}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    files = _find_prediction_files(month)
    if not files:
        os.makedirs(PSI_OUTDIR, exist_ok=True)
        payload = {"month": month, "status": "SKIPPED_NO_PREDICTION"}
        with open(os.path.join(PSI_OUTDIR, f"psi_{month}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    base = pd.read_parquet(baseline_path)
    print("BASE cols:", base.dtypes.to_dict())
    cur = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    print("CUR  cols:", cur.dtypes.to_dict())

    def _norm_names(df):
        df = df.rename(columns=lambda c: c.strip())
        return df

    base = _norm_names(base)
    cur  = _norm_names(cur)

        
    if "model_predictions" in cur.columns and "prediction" not in cur.columns:
        cur = cur.rename(columns={"model_predictions": "prediction"})

    if "score" in base.columns and "prediction" not in base.columns:
        base = base.rename(columns={"score": "prediction"})


    key_cols = ["score", "util_rate", "dti", "age"]
    cols = [c for c in key_cols if c in base.columns and c in cur.columns]
    if not cols:
        bn = base.select_dtypes(include="number").columns
        cn = cur.select_dtypes(include="number").columns
        cols = list(set(bn).intersection(set(cn)))[:5]

    psi_map = {}
    for c in cols:
        try:
            psi_map[c] = _psi_one_col(base[c].astype(float).values, cur[c].astype(float).values, bins=10)
        except Exception:
            continue

    overall = float(np.nanmax(list(psi_map.values())) if psi_map else np.nan)

    os.makedirs(PSI_OUTDIR, exist_ok=True)
    payload = {"month": month, "overall": overall, "psi": psi_map, "status": "OK"}
    with open(os.path.join(PSI_OUTDIR, f"psi_{month}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(PSI_HISTORY), exist_ok=True)
    row = pd.DataFrame([{
        "month": month, "overall": overall,
        "detail_json": json.dumps(psi_map, ensure_ascii=False)
    }])
    if os.path.exists(PSI_HISTORY):
        hist = pd.read_csv(PSI_HISTORY)
        hist = pd.concat([hist, row], ignore_index=True)
    else:
        hist = row
    hist.to_csv(PSI_HISTORY, index=False)

    print(f"PSI computed for {month}. overall={overall:.6f}, columns={list(psi_map.keys())}")
    return payload

def _cooldown_ok():
    meta = os.path.join(MODEL_BANK, "active.json")
    if not os.path.exists(meta):
        return True
    try:
        info = json.load(open(meta, "r"))
        until = info.get("cooldown_until")
        if not until:
            return True
        return datetime.utcnow() >= datetime.fromisoformat(until)
    except Exception:
        return True

def decide_branch_func(ti, **_):
    data = ti.xcom_pull(task_ids="monitor_compute_psi") or {}
    if data.get("status", "").startswith("SKIPPED"):
        return "skip_retrain"

    psi_map = data.get("psi", {}) or {}
    overall = float(data.get("overall", 0.0))

    bad1 = (overall >= PSI_THRESHOLD)
    bad_keys = sum(1 for v in psi_map.values() if v >= PSI_KEYCOL_THRESHOLD)
    bad2 = (bad_keys >= PSI_KEYCOL_MIN_COUNT)

    if (bad1 or bad2) and _cooldown_ok():
        return "retrain_model"
    return "skip_retrain"

def set_cooldown_func(**_):
    meta = os.path.join(MODEL_BANK, "active.json")
    info = {}
    if os.path.exists(meta):
        try:
            info = json.load(open(meta, "r"))
        except Exception:
            info = {}
    info["cooldown_until"] = (datetime.utcnow() + timedelta(days=RETRAIN_COOLDOWN_DAYS)).isoformat()
    os.makedirs(MODEL_BANK, exist_ok=True)
    json.dump(info, open(meta, "w"))
    print("Cooldown window has been set.")

# -----------------------------------------------------------------------------
# DAG Definition
# -----------------------------------------------------------------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 1)

with DAG(
    "dag_final",
    default_args=default_args,
    start_date=START_DATE,
    end_date=END_DATE,
    schedule_interval='0 0 1 * *',
    catchup=True,
    max_active_runs=1,
    tags=["credit-risk", "mlops"]
) as dag:

    # ---------------- Bronze
    bronze_loan_task = PythonOperator(task_id="bronze_loan", python_callable=bronze_loan_wrapper)
    bronze_clickstream_task = PythonOperator(task_id="bronze_clickstream", python_callable=bronze_clickstream_wrapper)
    bronze_attributes_task = PythonOperator(task_id="bronze_attributes", python_callable=bronze_attributes_wrapper)
    bronze_financials_task = PythonOperator(task_id="bronze_financials", python_callable=bronze_financials_wrapper)

    # ---------------- Silver
    silver_loan_task = PythonOperator(task_id="silver_loan", python_callable=silver_loan_wrapper)
    silver_clickstream_task = PythonOperator(task_id="silver_clickstream", python_callable=silver_clickstream_wrapper)
    silver_attributes_task = PythonOperator(task_id="silver_attributes", python_callable=silver_attributes_wrapper)
    silver_financials_task = PythonOperator(task_id="silver_financials", python_callable=silver_financials_wrapper)

    # ---------------- Gold
    gold_labels_task = PythonOperator(task_id="gold_labels", python_callable=gold_labels_wrapper)
    gold_engagement_task = PythonOperator(task_id="gold_engagement", python_callable=gold_engagement_wrapper)
    gold_cust_risk_task = PythonOperator(task_id="gold_customer_risk", python_callable=gold_cust_risk_wrapper)

    # ---------------- Training branch
    branch_initial_train = BranchPythonOperator(
        task_id="branch_initial_train_or_skip",
        python_callable=branch_initial_train_func,
    )
    skip_initial_train = EmptyOperator(task_id="skip_initial_train")
    train_task = PythonOperator(task_id="train_model", python_callable=train_model_wrapper)

    # activate_model_initial = PythonOperator(
    #     task_id='activate_model_initial',
    #     python_callable=activate_and_set_active_uri,
    #     op_kwargs={"source_task_id": "train_model"},
    #     trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
#)
    # ---------------- Inference
    # check_model = BranchPythonOperator(
    #     task_id="check_model_exists",
    #     python_callable=check_model_exists_func,
    # )

    # NOTE: Your utils/model_inference.py should accept --modelpath (file path).
    # If it only accepts a directory, change the script or wrap with a small shim.
    inference_scoring = BashOperator(
    task_id="inference_scoring",
    bash_command=(
        "set -euo pipefail; "
        "MODEL_URI='{{ ti.xcom_pull(task_ids=\"resolve_model_for_inference\", key=\"model_uri\") | default('', true) }}'; "
        "echo \"[INF] model_uri=$MODEL_URI\"; "
        "if [ -z \"$MODEL_URI\" ] || [ \"$MODEL_URI\" = 'None' ] || [ \"$MODEL_URI\" = 'null' ]; then "
        "  echo '[INF] No valid model — skipping inference.'; "
        "  exit 0; "
        "fi; "
        "ls -l \"$(dirname \"$MODEL_URI\")\" || true; "
        "cd /opt/airflow && "
        "python utils/model_inference.py "
        "--snapshotdate {{ ds }} "
        "--modelname \"$MODEL_URI\" "
    ),
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    resolve_model = PythonOperator(
    task_id="resolve_model_for_inference",
    python_callable=resolve_model_for_inference,
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,  # 任一上游成功即可
    )


    # ---------------- PSI
    compute_psi = PythonOperator(
        task_id="monitor_compute_psi",
        python_callable=compute_psi_func,
        op_kwargs={"ds": "{{ ds }}"},
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # ---------------- Retrain branch
    branch_retrain = BranchPythonOperator(
        task_id="branch_retrain_or_skip",
        python_callable=decide_branch_func,
    )
    retrain_task = PythonOperator(task_id="retrain_model", python_callable=train_model_wrapper)
    activate_model_retrain = PythonOperator(
        task_id='activate_model_retrain',
        python_callable=activate_and_set_active_uri,
        op_kwargs={"source_task_id": "retrain_model"},
    )
    skip_retrain = EmptyOperator(task_id="skip_retrain")
    set_cooldown = PythonOperator(task_id="set_retrain_cooldown", python_callable=set_cooldown_func)
    done = EmptyOperator(task_id="done")
    
    inference_scoring.trigger_rule = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    compute_psi.trigger_rule       = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    done.trigger_rule              = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    # ---------------- Dependencies
    #Bronze -> Silver
    bronze_loan_task >> silver_loan_task
    bronze_clickstream_task >> silver_clickstream_task
    bronze_attributes_task >> silver_attributes_task
    bronze_financials_task >> silver_financials_task

    # Silver -> Gold
    silver_loan_task >> gold_labels_task
    silver_clickstream_task >> gold_engagement_task
    silver_financials_task >> gold_cust_risk_task

    # Features/labels ready -> initial train branch
    [silver_attributes_task, silver_financials_task, gold_labels_task] >> branch_initial_train
    branch_initial_train >> train_task
    branch_initial_train >> skip_initial_train


    # After train: only go to activation if we have a new model; otherwise go to check
    # 分支1：有新模型 → activate
    #train_task >> activate_model_initial
    #train_task >> inference_scoring
    # 分支2：沒新模型 → check
    #skip_initial_train >> check_model
    #skip_initial_train >> resolve_model

    [train_task, skip_initial_train] >> resolve_model
    resolve_model >> inference_scoring

    # 兩條路會和在 resolver（允許另一條被 SKIP）
    #[activate_model_initial, check_model] >> resolve_model
    # 匯流點（要允許一邊被 skipped）
    resolve_model >> inference_scoring
    #[activate_model_initial, check_model] >>  inference_scoring
    inference_scoring >> compute_psi
    compute_psi >> branch_retrain
    branch_retrain >> retrain_task >> activate_model_retrain >> set_cooldown >> done
    branch_retrain >> skip_retrain >> done
