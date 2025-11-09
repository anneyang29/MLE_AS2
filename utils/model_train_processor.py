import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler


# ---------- SafeOrdinalEncoder ----------
class SafeOrdinalEncoder:
    """Ordinal encoder safe for unseen and NaN values."""

    def __init__(self, unknown_token="__UNK__", unknown_value=-1):
        self.unknown_token = unknown_token
        self.unknown_value = unknown_value
        self.mapping_ = None

    def fit(self, series: pd.Series):
        s = series.astype(str).fillna(self.unknown_token)
        cats = sorted(s.unique().tolist())
        if self.unknown_token not in cats:
            cats.append(self.unknown_token)
        self.mapping_ = {v: i for i, v in enumerate(cats)}
        self.mapping_[self.unknown_token] = self.unknown_value
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        if self.mapping_ is None:
            raise RuntimeError("SafeOrdinalEncoder is not fitted.")
        s = series.astype(str).fillna(self.unknown_token)
        return s.map(lambda x: self.mapping_.get(x, self.unknown_value)).astype("int32")

    def fit_transform(self, series: pd.Series) -> pd.Series:
        return self.fit(series).transform(series)

    @property
    def classes_(self):
        if self.mapping_ is None:
            return None
        return [k for k, v in self.mapping_.items() if v != self.unknown_value]


# ---------- Main Processor ----------
@dataclass
class DataProcessor:
    feature_order: list
    skewed_cols: list = field(
        default_factory=lambda: [
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Delay_from_due_date",
            "Outstanding_Debt",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance",
            "Loans_per_Credit_Item",
            "Debt_to_Salary",
            "EMI_to_Salary",
            "Repayment_Ability",
            "Loan_Extent",
            "Debt_to_Income"
        ]
    )
    num_impute_values: dict = field(default_factory=dict)
    scaler: StandardScaler = None
    occ_encoder: SafeOrdinalEncoder = None
    pb_mapping: dict = field(default_factory=lambda: {str(i): i for i in range(6)})
    pb_unknown_value: int = -1

    # ---------- internal helpers ----------
    def _fill_missing_train(self, df, cat_cols, num_cols):
        for c in cat_cols:
            df[c] = df[c].astype("object").fillna("__UNK__")
        for c in num_cols:
            med = pd.to_numeric(df[c], errors="coerce").median()
            self.num_impute_values[c] = med
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med)

    def _fill_missing_infer(self, df, cat_cols, num_cols):
        for c in cat_cols:
            df[c] = df[c].astype("object").fillna("__UNK__")
        for c in num_cols:
            med = self.num_impute_values.get(c, np.nan)
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med)

    def _encode_payment_behaviour(self, s):
        return s.astype(str).map(lambda x: self.pb_mapping.get(x, self.pb_unknown_value)).astype("int32")

    def _add_debt_to_income(self, X):
        """Add Debt_to_Income derived feature."""
        if "Debt_to_Income" not in X.columns:
            X["Debt_to_Income"] = X["Outstanding_Debt"] / X["Annual_Income"]

    # ---------- main fit ----------
    def fit(self, X_train: pd.DataFrame):
        X_train = X_train.copy()

        # 1. Add derived features
        self._add_debt_to_income(X_train)

        # 2. column grouping
        cat_cols = [c for c in ["Occupation", "Payment_Behaviour"] if c in self.feature_order]
        num_cols = [c for c in self.feature_order if c not in cat_cols]

        # 3. missing
        self._fill_missing_train(X_train, cat_cols, num_cols)

        # 4. categorical encoding
        if "Payment_Behaviour" in cat_cols:
            X_train["Payment_Behaviour"] = self._encode_payment_behaviour(X_train["Payment_Behaviour"])

        if "Occupation" in cat_cols:
            self.occ_encoder = SafeOrdinalEncoder().fit(X_train["Occupation"])
            X_train["Occupation"] = self.occ_encoder.transform(X_train["Occupation"])

        # 5. log transform skewed numeric columns
        for c in self.skewed_cols:
            if c in num_cols:
                X_train[c] = np.log1p(np.clip(pd.to_numeric(X_train[c], errors="coerce"), 0, None))

        # 6. scale
        self.scaler = StandardScaler().fit(X_train[self.feature_order])

        return self

    # ---------- main transform ----------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Add derived features
        self._add_debt_to_income(X)

        cat_cols = [c for c in ["Occupation", "Payment_Behaviour"] if c in self.feature_order]
        num_cols = [c for c in self.feature_order if c not in cat_cols]

        self._fill_missing_infer(X, cat_cols, num_cols)

        if "Payment_Behaviour" in cat_cols:
            X["Payment_Behaviour"] = self._encode_payment_behaviour(X["Payment_Behaviour"])

        if "Occupation" in cat_cols and self.occ_encoder:
            X["Occupation"] = self.occ_encoder.transform(X["Occupation"])

        for c in self.skewed_cols:
            if c in num_cols:
                X[c] = np.log1p(np.clip(pd.to_numeric(X[c], errors="coerce"), 0, None))

        X_scaled = pd.DataFrame(
            self.scaler.transform(X[self.feature_order]),
            columns=self.feature_order,
            index=X.index,
        )
        return X_scaled