# -*- coding: utf-8 -*-
"""
Model training script (Spark + Pandas + XGBoost)
- 使用自訂 DataProcessor（來自 utils/model_train_processor.py）：清洗、穩定編碼
- 以相對路徑讀取 Parquet（與 Airflow volume 掛載對齊）
- 產出 model_bank/credit_model_YYYY_MM_DD.pkl
"""

import argparse
import os
import glob
import pprint
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import xgboost as xgb

# 你的專案處理器
from utils.model_train_processor import DataProcessor  # 內含你自訂的編碼/清洗


# ----------------------------
# 主程式
# ----------------------------
def main(snapshotdate: str):
    print("\n\n--- starting job ---\n")

    # Init Spark
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("model_train_main")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # --------------- Config ---------------
        model_train_date_str = snapshotdate
        train_test_period_months = 12
        oot_period_months = 2
        train_test_ratio = 0.8

        config = {}
        config["model_train_date_str"] = model_train_date_str
        config["train_test_period_months"] = train_test_period_months
        config["oot_period_months"] = oot_period_months
        config["model_train_date"] = datetime.strptime(model_train_date_str, "%Y-%m-%d")
        config["oot_end_date"] = config["model_train_date"] - timedelta(days=1)
        config["oot_start_date"] = config["model_train_date"] - relativedelta(months=oot_period_months)
        config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
        config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=train_test_period_months)
        config["train_test_ratio"] = train_test_ratio

        pprint.pprint(config)

        # --------------- Load label store ---------------
        # gold/label_store/ 讀全部並在 Spark 做期間過濾
        folder_path = "datamart/gold/label_store/"
        files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, "*"))]
        if not files_list:
            raise FileNotFoundError(f"No label_store parquet found in {folder_path}")

        label_store_sdf = spark.read.parquet(*files_list)
        print("label_store row_count:", label_store_sdf.count())

        # 只保留必要欄位（避免 toPandas 記憶體爆）
        # 假設 label 欄名為 'label'，以及一定含有 Customer_ID、snapshot_date
        base_label_cols = [c for c in ["Customer_ID", "snapshot_date", "label", "loan_id", "label_def"] if c in label_store_sdf.columns]
        label_store_sdf = label_store_sdf.select(*base_label_cols)

        labels_sdf = label_store_sdf.filter(
            (col("snapshot_date") >= F.lit(config["train_test_start_date"])) &
            (col("snapshot_date") <= F.lit(config["oot_end_date"]))
        )
        print("extracted labels_sdf rows:", labels_sdf.count())
        print("train_test_start_date:", config["train_test_start_date"])
        print("oot_end_date:", config["oot_end_date"])

        # --------------- Load features (silver) ---------------
        # attributes
        attr_path = "datamart/silver/attr/"
        attr_files = [attr_path + os.path.basename(f) for f in glob.glob(os.path.join(attr_path, "*"))]
        if not attr_files:
            raise FileNotFoundError(f"No attributes parquet found in {attr_path}")
        attributes_sdf = spark.read.parquet(*attr_files)
        print("silver_attributes row_count:", attributes_sdf.count())

        attributes_cols = [c for c in ["Customer_ID", "snapshot_date", "Age", "Occupation"] if c in attributes_sdf.columns]
        attributes_sdf_subset = (
            attributes_sdf.select(*attributes_cols)
            .filter(col("snapshot_date") <= F.lit(config["oot_end_date"]))
        )
        # Drop snapshot_date to avoid ambiguity in single-key join
        attributes_sdf_subset = attributes_sdf_subset.drop("snapshot_date")
        attributes_sdf_subset.show(5)

        # financials
        fin_path = "datamart/silver/fin/"
        fin_files = [fin_path + os.path.basename(f) for f in glob.glob(os.path.join(fin_path, "*"))]
        if not fin_files:
            raise FileNotFoundError(f"No financials parquet found in {fin_path}")
        financials_sdf = spark.read.parquet(*fin_files)
        print("silver_financials row_count:", financials_sdf.count())

        financials_cols = [
            "Customer_ID", "snapshot_date", "Annual_Income", "Monthly_Inhand_Salary",
            "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
            "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
            "Credit_Mix", "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
            "Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance",
            "Num_Fin_Pdts", "Loans_per_Credit_Item", "Debt_to_Salary", "EMI_to_Salary",
            "Repayment_Ability", "Loan_Extent"
        ]
        fin_keep = [c for c in financials_cols if c in financials_sdf.columns]
        financials_sdf_subset = (
            financials_sdf.select(*fin_keep)
            .filter(col("snapshot_date") <= F.lit(config["oot_end_date"]))
        )
        # Drop snapshot_date to avoid ambiguity in single-key join
        financials_sdf_subset = financials_sdf_subset.drop("snapshot_date")
        financials_sdf_subset.show(5)

        # --------------- Assemble training table (single key join) ---------------
        merged_sdf = labels_sdf.select([col(c) for c in labels_sdf.columns])  # fresh copy
        merged_sdf = merged_sdf.join(attributes_sdf_subset, on="Customer_ID", how="left")
        merged_sdf = merged_sdf.join(financials_sdf_subset, on="Customer_ID", how="left")
        print("merged_sdf row_count:", merged_sdf.count())
        print("merged_sdf columns:", merged_sdf.columns)

        # 只保留訓練需要欄位
        feature_cols = [
            "Age", "Occupation", "Annual_Income",
            "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
            "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
            "Num_of_Delayed_Payment", "Changed_Credit_Limit",
            "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
            "Credit_Utilization_Ratio", "Credit_History_Age", "Total_EMI_per_month",
            "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance",
            "Num_Fin_Pdts", "Loans_per_Credit_Item", "Debt_to_Salary",
            "EMI_to_Salary", "Repayment_Ability", "Loan_Extent"
        ]
        # 保底：部分欄位可能不存在，動態篩选
        keep_cols = ["Customer_ID", "snapshot_date", "label"] + [c for c in feature_cols if c in merged_sdf.columns]
        merged_sdf = merged_sdf.select(*keep_cols)

        # --------------- toPandas + 時間型別一致化 ---------------
        merged_df = merged_sdf.toPandas()
        merged_df["snapshot_date"] = pd.to_datetime(merged_df["snapshot_date"], errors="coerce")

        # 切分 train/test 與 oot（用 pandas 的 Timestamp）
        oot_mask = (merged_df["snapshot_date"] >= pd.Timestamp(config["oot_start_date"])) & \
                   (merged_df["snapshot_date"] <= pd.Timestamp(config["oot_end_date"]))
        tt_mask = (merged_df["snapshot_date"] >= pd.Timestamp(config["train_test_start_date"])) & \
                  (merged_df["snapshot_date"] <= pd.Timestamp(config["train_test_end_date"]))

        oot_pdf = merged_df.loc[oot_mask].copy()
        train_test_pdf = merged_df.loc[tt_mask].copy()

        # 取實際存在的特徵欄
        feature_cols_kept = [c for c in feature_cols if c in train_test_pdf.columns]

        # --------------- Train/Test split（含安全檢查） ---------------
        if train_test_pdf["label"].nunique() < 2:
            raise ValueError("Train/Test 期間內的 label 只有單一類別，無法訓練。請調整期間或檢查標記。")

        stratify_vec = train_test_pdf["label"] if train_test_pdf["label"].nunique() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            train_test_pdf[feature_cols_kept],
            train_test_pdf["label"],
            test_size=1 - config["train_test_ratio"],
            random_state=88,
            shuffle=True,
            stratify=stratify_vec
        )

        X_oot = oot_pdf[feature_cols_kept]
        y_oot = oot_pdf["label"] if "label" in oot_pdf.columns else pd.Series([], dtype=int)

        print("X_train", X_train.shape[0])
        print("X_test", X_test.shape[0])
        print("X_oot", X_oot.shape[0])
        print("y_train", y_train.shape[0], round(float(y_train.mean()), 4))
        print("y_test", y_test.shape[0], round(float(y_test.mean()), 4))
        if len(y_oot) > 0:
            print("y_oot", y_oot.shape[0], round(float(y_oot.mean()), 4))
        else:
            print("y_oot is empty (no OOT labels in the specified period).")

        # --------------- Data Processing（僅 DataProcessor） ---------------
        # Pass the initial feature list to the processor
        processor = DataProcessor(feature_order=feature_cols_kept)
        processor.fit(X_train)

        # Get the final feature list from the processor AFTER fitting
        final_feature_cols = processor.feature_order

        X_train_fe = processor.transform(X_train)
        X_test_fe = processor.transform(X_test)
        X_oot_fe = processor.transform(X_oot) if len(X_oot) > 0 else X_oot

        # --------------- Train Model ---------------
        # Define the hyperparameter space to search
        param_dist = {
            'n_estimators': [25, 50],
            'max_depth': [2, 3],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8],
            'gamma': [0, 0.1],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }

        # Create a scorer based on AUC score
        auc_scorer = make_scorer(roc_auc_score)

        # Create base model
        xgb_clf = xgb.XGBClassifier(eval_metric="logloss", random_state=42)

        # Set up the random search with cross-validation
        random_search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=param_dist,
            scoring=auc_scorer,
            n_iter=100,
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )

        # Perform the random search
        random_search.fit(X_train_fe, y_train)
        best_model = random_search.best_estimator_

        # --------------- Evaluate ---------------
        y_train_proba = best_model.predict_proba(X_train_fe)[:, 1]
        y_test_proba = best_model.predict_proba(X_test_fe)[:, 1]
        train_auc_score = roc_auc_score(y_train, y_train_proba)
        test_auc_score = roc_auc_score(y_test, y_test_proba)

        if len(y_oot) >= 2 and y_oot.nunique() == 2:
            y_oot_proba = best_model.predict_proba(X_oot_fe)[:, 1]
            oot_auc_score = roc_auc_score(y_oot, y_oot_proba)
        else:
            oot_auc_score = np.nan

        print("Model Performance:")
        print(f"Train AUC: {train_auc_score:.4f} | GINI: {2 * train_auc_score - 1:.4f}")
        print(f"Test  AUC: {test_auc_score:.4f}  | GINI: {2 * test_auc_score - 1:.4f}")
        print(f"OOT   AUC: {oot_auc_score:.4f}   | GINI: {2 * oot_auc_score - 1:.4f}")

        # --------------- Build artefact ---------------
        model_artefact = {
            "model": best_model,
            "model_version": "credit_model_" + config["model_train_date_str"].replace("-", "_"),
            "preprocessing_transformers": {
                "data_processor": processor
            },
            "data_dates": config,
            "data_stats": {
                "X_train": int(X_train.shape[0]),
                "X_test": int(X_test.shape[0]),
                "X_oot": int(X_oot.shape[0]),
                "y_train": round(float(y_train.mean()), 4),
                "y_test": round(float(y_test.mean()), 4),
                "y_oot": round(float(y_oot.mean()), 4) if len(y_oot) > 0 else None
            },
            "results": {
                "auc_train": float(train_auc_score),
                "auc_test": float(test_auc_score),
                "auc_oot": float(oot_auc_score) if not np.isnan(oot_auc_score) else None,
                "gini_train": round(2 * train_auc_score - 1, 4),
                "gini_test": round(2 * test_auc_score - 1, 4),
                "gini_oot": round(2 * oot_auc_score - 1, 4) if not np.isnan(oot_auc_score) else None
            },
            "hp_params": best_model.get_params()
        }
        pprint.pprint(model_artefact)

        # --------------- Save artefact ---------------
        model_bank_directory = "model_bank/"
        os.makedirs(model_bank_directory, exist_ok=True)
        file_path = os.path.join(model_bank_directory, model_artefact["model_version"] + ".pkl")
        with open(file_path, "wb") as f:
            pickle.dump(model_artefact, f)
        print(f"Model saved to {file_path}")

        with open("best model name.txt", "w") as f:
            f.write(file_path)

        with open(file_path, "rb") as f:
            loaded = pickle.load(f)
        test_reload_auc = roc_auc_score(y_test, loaded["model"].predict_proba(X_test_fe)[:, 1])
        print("Reloaded TEST AUC score:", test_reload_auc)
        print("Model loaded successfully!")

        print("\n--- Creating baseline predictions for monitoring ---")
        try:
            baseline_date_obj = config["train_test_end_date"].replace(day=1)
            baseline_date_str = baseline_date_obj.strftime('%Y_%m_%d')
            print(f"Using data from {baseline_date_obj.strftime('%Y-%m-%d')} for baseline.")

            # --- Load data ---
            folder_path = "datamart/silver/attr/"
            attributes_sdf = spark.read.parquet(folder_path + 'silver_attr_mthly_' + baseline_date_str + '.parquet')
            attributes_cols = [c for c in ["Customer_ID", "snapshot_date", "Age", "Occupation"] if c in attributes_sdf.columns]
            baseline_attr_sdf = attributes_sdf[attributes_cols]
            print("attributes row_count:", baseline_attr_sdf.count())

            folder_path = "datamart/silver/fin/"
            financials_sdf = spark.read.parquet(folder_path + 'silver_fin_mthly_' + baseline_date_str + '.parquet')
            financials_cols = [
                "Customer_ID", "Annual_Income", "Monthly_Inhand_Salary",
                "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
                "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit", 
                "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt", 
                "Credit_Utilization_Ratio", "Credit_History_Age", "Total_EMI_per_month", 
                "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance",
                "Num_Fin_Pdts", "Loans_per_Credit_Item", "Debt_to_Salary", 
                "EMI_to_Salary", "Repayment_Ability", "Loan_Extent"
            ]
            fin_keep = [c for c in financials_cols if c in financials_sdf.columns]
            baseline_fin_sdf = financials_sdf[fin_keep]
            print("financials row_count:", baseline_fin_sdf.count())

            # --- Merge ---
            baseline_merged_sdf = baseline_attr_sdf.select([col(c) for c in baseline_attr_sdf.columns])
            baseline_merged_sdf = baseline_merged_sdf.join(baseline_fin_sdf, on="Customer_ID", how="inner")
            print(f"merged_df row_count: {baseline_merged_sdf.count()}")

            # --- Convert to Pandas ---
            baseline_merged_pdf = baseline_merged_sdf.toPandas()
            customer_ids = baseline_merged_pdf["Customer_ID"].copy()
            snapshot_dates = baseline_merged_pdf["snapshot_date"].copy()
            baseline_pdf_clean = baseline_merged_pdf.drop(columns=['Customer_ID', 'snapshot_date'])

            # --- Preprocess and Predict ---
            processor = model_artefact['preprocessing_transformers']['data_processor']
            baseline_features_fe = processor.transform(baseline_pdf_clean)
            model = model_artefact["model"]
            baseline_predictions = model.predict_proba(baseline_features_fe)[:, 1]
            print(baseline_predictions[:5])

            # --- 建立純資料 DataFrame ---
            output_pdf = pd.DataFrame({
                "Customer_ID": customer_ids.values,
                "snapshot_date": snapshot_dates.values,
                "model_name": model_artefact["model_version"] + '.pkl',
                "prediction": baseline_predictions
            })
            
            # === 關鍵：先寫 CSV（不需要序列化） ===
            partition_name = model_artefact['model_version'] + "_psi_ref_preds"
            csv_temp_path = os.path.join(model_bank_directory, partition_name + "_temp.csv")
            output_pdf.to_csv(csv_temp_path, index=False)
            print(f'Temporary CSV saved to: {csv_temp_path}')
            
            # === 用 Spark 讀取 CSV 並寫成 Parquet ===
            parquet_path = os.path.join(model_bank_directory, partition_name + ".parquet")
            read_spark = spark.read.csv(csv_temp_path, header=True, inferSchema=True)
            read_spark.show(5)
            read_spark.write.mode("overwrite").parquet(parquet_path)
            
            print(f'PSI baseline predictions saved to: {parquet_path}')
            
            # 清除臨時 CSV
            os.remove(csv_temp_path)
            print(f'Temporary CSV removed.')

        except FileNotFoundError as e:
            print(f"ERROR: Could not create baseline. {e}")
        except Exception as e:
            print(f"An unexpected error occurred during baseline creation: {e}")
            import traceback
            traceback.print_exc()


        print("\n--- completed job ---\n")

    finally:
        print("Stopping Spark session...")
        spark.stop()


        # --------------- Save baseline predictions (for output drift monitoring) ---------------
        # print("\n--- Creating baseline predictions for monitoring ---")
        # try:
        #     # Determine the correct baseline date (first day of the last month of training)
        #     baseline_date_obj = config["train_test_end_date"].replace(day=1)
        #     baseline_date_str = baseline_date_obj.strftime('%Y_%m_%d')

        #     print(f"Using data from {baseline_date_obj.strftime('%Y-%m-%d')} for baseline.")

        #     # --- Load the specific monthly data for the baseline ---
        #     # Attributes
        #     folder_path = "datamart/silver/attr/"
        #     attributes_sdf = spark.read.parquet(folder_path + 'silver_attr_mthly_' + baseline_date_str + '.parquet')
        #     attributes_cols = [c for c in ["Customer_ID", "snapshot_date", "Age", "Occupation"] if c in attributes_sdf.columns]
        #     baseline_attr_sdf = attributes_sdf[attributes_cols]
        #     print("attributes row_count:", baseline_attr_sdf.count())

        #     # Financials
        #     folder_path = "datamart/silver/fin/"
        #     financials_sdf = spark.read.parquet(folder_path + 'silver_fin_mthly_' + baseline_date_str + '.parquet')
        #     financials_cols = [
        #     "Customer_ID", "Annual_Income", "Monthly_Inhand_Salary",
        #     "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
        #     "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
        #     "Credit_Mix", "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
        #     "Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance",
        #     "Num_Fin_Pdts", "Loans_per_Credit_Item", "Debt_to_Salary", "EMI_to_Salary",
        #     "Repayment_Ability", "Loan_Extent"
        # ]
        #     fin_keep = [c for c in financials_cols if c in financials_sdf.columns]
        #     baseline_fin_sdf = financials_sdf[fin_keep]
        #     print("financials row_count:", baseline_fin_sdf.count())

        #     # --- Merge and Prepare Data ---
        #     # Drop snapshot_date from the financials table to avoid duplicate columns after join
        #     baseline_merged_sdf = baseline_attr_sdf.select([col(c) for c in baseline_attr_sdf.columns])
        #     baseline_merged_sdf = baseline_merged_sdf.join(baseline_fin_sdf, on="Customer_ID", how="inner")
        #     print(f"merged_df row_count: {baseline_merged_sdf.count()}")

        #     # Convert to Pandas
        #     baseline_merged_sdf = baseline_merged_sdf.toPandas()
        #     customer_ids = baseline_merged_sdf["Customer_ID"].copy()
        #     snapshot_dates = baseline_merged_sdf["snapshot_date"].copy()
        #     baseline_pdf_clean = baseline_merged_sdf.drop(columns=['Customer_ID', 'snapshot_date'])

        #     # --- Preprocess using the same processor from the artifact ---
        #     # The 'processor' object is already fitted and available from the training step
        #     processor = model_artefact['preprocessing_transformers']['data_processor']
        #     baseline_features_fe = processor.transform(baseline_pdf_clean)


        #     # --- Predict using the newly trained model ---
        #     model = model_artefact["model"]
        #     baseline_predictions = model.predict_proba(baseline_features_fe)[:, 1]

            # --- Prepare and Save Output ---
        #     output_pdf = baseline_merged_sdf[["Customer_ID", "snapshot_date"]].copy()
        #     output_pdf["model_name"] = model_artefact["model_version"] + '.pkl'
        #     output_pdf["prediction"] = baseline_predictions
            
        #     partition_name = model_artefact['model_version'] + "_psi_ref_preds" + '.parquet'
        #     filepath = os.path.join(model_bank_directory, partition_name)

        #     spark.createDataFrame(output_pdf).write.mode("overwrite").parquet(filepath)
        #     print('PSI baseline predictions saved to:', filepath)

        # except FileNotFoundError as e:
        #     print(f"ERROR: Could not create baseline. {e}")
        # except Exception as e:
        #     print(f"An unexpected error occurred during baseline creation: {e}")
             # --- 建立純資料 DataFrame ---
    #         output_pdf = pd.DataFrame({
    #             "Customer_ID": customer_ids.values,
    #             "snapshot_date": snapshot_dates.values,
    #             "model_name": model_artefact["model_version"] + '.pkl',
    #             "prediction": baseline_predictions
    #         })
            
            
    #         # === 關鍵：先寫 CSV（不需要序列化） ===
    #         partition_name = model_artefact['model_version'] + "_psi_ref_preds"
    #         csv_temp_path = os.path.join(model_bank_directory, partition_name + "_temp.csv")
    #         output_pdf.to_csv(csv_temp_path, index=False)
    #         print(f'Temporary CSV saved to: {csv_temp_path}')


            
    #         # === 用 Spark 讀取 CSV 並寫成 Parquet ===
    #         parquet_path = os.path.join(model_bank_directory, partition_name + ".parquet")
    #         read_spark = spark.read.csv(csv_temp_path, header=True, inferSchema=True)
    #         read_spark.write.mode("overwrite").parquet(parquet_path)
    #         read_spark.show(5)

    #         print(f'PSI baseline predictions saved to: {parquet_path}')
            
    #         # 清除臨時 CSV
    #         os.remove(csv_temp_path)
    #         print(f'Temporary CSV removed.')

    #     except FileNotFoundError as e:
    #         print(f"ERROR: Could not create baseline. {e}")
    #     except Exception as e:
    #         print(f"An unexpected error occurred during baseline creation: {e}")
    #         import traceback
    #         traceback.print_exc()

    #     print("\n--- completed job ---\n")

    # finally:
    #     print("Stopping Spark session...")
    #     spark.stop()

      


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
