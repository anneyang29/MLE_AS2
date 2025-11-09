# model_inference.py
# -*- coding: utf-8 -*-
"""
Usage:
  python model_inference.py --snapshotdate "2024-09-01" --modelname "credit_model_2024_09_01.pkl"

Notes:
- 不需要 OneHotEncoder；前處理器已將類別欄位數值化並標準化（訓練時已儲存於 artefact 的 data_processor）
- 會從 model_bank/<modelname> 載入 artefact，其中包含:
    - model
    - preprocessing_transformers: {"data_processor": ...}
- 讀取當月的 silver_attr_mthly_YYYY_MM_01.parquet 與 silver_fin_mthly_YYYY_MM_01.parquet
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pprint
import pickle

import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType

from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, roc_auc_score

import xgboost as xgb

# 前處理直接使用 artefact 內保存的 data_processor


# to call this script: python model_inference.py --snapshotdate "2024-09-01" --modelname "credit_model_2024_09_01.pkl"

def main(snapshotdate, modelname):
    print('\n---starting job---\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("model_inference") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    try: 
        # --- set up config ---
        config = {}
        config["snapshot_date_str"] = snapshotdate
        config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
        config["model_name"] = modelname
        config["model_bank_directory"] = "model_bank/"
        config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
        pprint.pprint(config)
        
        # --- load model artefact from model bank ---
        with open(config["model_artefact_filepath"], 'rb') as file:
            model_artefact = pickle.load(file)
        print("Model loaded successfully! " + config["model_artefact_filepath"])

        # 抽取模型與前處理器（你訓練時已存進 artefact 的 data_processor）
        model = model_artefact["model"]
        transformer_processor = model_artefact["preprocessing_transformers"]["data_processor"]
    
        # --- load feature store ---
        # connect to silver attributes table
        folder_path = "datamart/silver/attr/"
        attributes_path = folder_path + 'silver_attr_mthly_' + config["snapshot_date_str"].replace("-", "_") + '.parquet'
        attributes_sdf = spark.read.parquet(attributes_path)
        
        # take only important features
        attributes_cols = ['Customer_ID', 'Age', 'Occupation', 'snapshot_date']
        # 若部分欄位不存在於該月檔，先用安全的方式取用
        attributes_cols_present = [c for c in attributes_cols if c in attributes_sdf.columns]
        attributes_sdf_subset = attributes_sdf[attributes_cols_present]
        print("attributes row_count:", attributes_sdf_subset.count())
        
        # connect to silver financials table
        folder_path = "datamart/silver/fin/"
        financials_path = folder_path + 'silver_fin_mthly_' + config["snapshot_date_str"].replace("-", "_") + '.parquet'
        financials_sdf = spark.read.parquet(financials_path)
        
        # take only important features
        financials_cols = [
            'Customer_ID', 'Annual_Income', 'Monthly_Inhand_Salary',
            'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
            'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
            'Credit_Mix', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
            'Total_EMI_per_month', 'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance',
            'Num_Fin_Pdts', 'Loans_per_Credit_Item', 'Debt_to_Salary', 'EMI_to_Salary', 'Repayment_Ability', 'Loan_Extent'
        ]
        financials_cols_present = [c for c in financials_cols if c in financials_sdf.columns]
        financials_sdf_subset = financials_sdf[financials_cols_present]
        print("financials row_count:", financials_sdf_subset.count())
    
        # --- preprocess data for modeling ---
        # Merge attributes and financials into 1 table (no labels at this point in time)
        # use inner join coz all customer ID records must have all features from both tables to make inference
        merged_df = attributes_sdf_subset.select([col(c) for c in attributes_sdf_subset.columns])  # make a fresh copy
        merged_df = merged_df.join(financials_sdf_subset, on="Customer_ID", how="inner")
        
        # Check size of resultant table. 
        print(f"merged_df row_count: {merged_df.count()}")
        
        # Convert to Python pandas, prepare data for modeling
        merged_df = merged_df.toPandas()
        
        # After merging successfully, remove Customer_ID / snapshot_date（非特徵）
        merged_df_clean = merged_df.drop(columns=['Customer_ID', 'snapshot_date'], errors='ignore')
        
        # Apply data processing steps from saved transformer（不做 one-hot，直接用訓練時的 data_processor）
        X_inference = transformer_processor.transform(merged_df_clean)
        print('X_inference rows: ', X_inference.shape[0])
    
        # --- model prediction inference ---
        # predict model
        y_inference = model.predict_proba(X_inference)[:, 1]
        
        # prepare output（保留識別欄位以便追蹤）
        id_cols = [c for c in ["Customer_ID", "snapshot_date"] if c in merged_df.columns]
        y_inference_pdf = merged_df[id_cols].copy()
        y_inference_pdf["model_name"] = config["model_name"]
        y_inference_pdf["model_predictions"] = y_inference.astype(float)
        
        # snapshot_date 正規化成 YYYY-MM-DD 字串，避免 schema 差異
        if "snapshot_date" in y_inference_pdf.columns:
            y_inference_pdf["snapshot_date"] = pd.to_datetime(y_inference_pdf["snapshot_date"]).dt.strftime("%Y-%m-%d")
    
        # --- save model inference to datamart gold table ---
        gold_directory = f"datamart/gold/model_predictions/{config['model_name'][:-4]}/"
        print(gold_directory)
        if not os.path.exists(gold_directory):
            os.makedirs(gold_directory)
        
        partition_name = config["model_name"][:-4] + "_preds_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
        filepath = partition_name
        
        # Convert pandas df to spark df and write to parquet
        spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath)

        # pdf = y_inference_pdf.copy()
        # pdf = pdf.replace([np.inf, -np.inf], np.nan)

        # obj_cols = [c for c in pdf.columns if pdf[c].dtype == "object"]
        # for c in obj_cols:
        #     pdf[c] = pdf[c].astype(str)
        
        # sdf = spark.createDataFrame(pdf)

        # from pyspark.sql import functions as F
        
        # sdf = sdf.select([F.col(c) for c in sdf.columns])
        # sdf = sdf.coalesce(1)

        # out_abs = os.path.abspath(filepath).replace("\\", "/")
        # out_uri = "file:///" + out_abs.lstrip("/")

        # sdf.write.mode("overwrite").parquet(out_uri)

        # print('saved to:', out_uri)

    finally:
        print("Stopping Spark session...")
        spark.stop()
        print('\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    args = parser.parse_args()
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
