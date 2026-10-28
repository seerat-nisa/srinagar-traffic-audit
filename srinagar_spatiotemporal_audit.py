"""
Post-Publication Spatio-Temporal Audit for Srinagar Smart City Traffic Logs.
Audits (1) regime-transition sensitivity in CDAC stage timing logs and
(2) pre-split aggregation leakage in multi-corridor ANPR vehicle event logs.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR


def audit_signal_timing(filepath: str) -> None:
    print("=== 1. CDAC SIGNAL TIMING AUDIT ===")
    df = pd.read_excel(filepath)
    df = df.replace("FLASH", np.nan).fillna(0).dropna()

    X = df[["ALLOCATED GREEN", "AVAILABLE GREEN"]]
    y = df["UTILIZED GREEN"].astype(float)

    # 1A. Original Random Split (reproducing published baseline)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    for name, model in [
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(random_state=42)),
        ("RBF-SVR (eps=0.1)", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    ]:
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        print(
            f"[Random 80/20] {name:18s} | MSE: {mean_squared_error(y_te, preds):8.4f} "
            f"| MAE: {mean_absolute_error(y_te, preds):6.4f} | R2: {r2_score(y_te, preds):8.4f}"
        )

    # 1B. Chronological Split (train on first 80% of day, test on final 20% including 21:00 shutdown)
    split_idx = int(len(df) * 0.8)
    X_tr_c, X_te_c = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr_c, y_te_c = y.iloc[:split_idx], y.iloc[split_idx:]
    svr = SVR(kernel="rbf", C=1.0, epsilon=0.1).fit(X_tr_c, y_tr_c)
    preds_c = svr.predict(X_te_c)
    regime_mask = y_te_c != 12.0
    print(
        f"[Chronological 80/20] RBF-SVR Overall MAE: {mean_absolute_error(y_te_c, preds_c):.4f}s | "
        f"Regime-Transition MAE: {mean_absolute_error(y_te_c[regime_mask], preds_c[regime_mask]):.4f}s\n"
    )


def audit_volume_pipeline(filepath: str) -> None:
    print("=== 2. ANPR TRAFFIC VOLUME LEAKAGE AUDIT ===")
    df = pd.read_excel(filepath)
    df["Date and Time"] = pd.to_datetime(df["Date and Time"])
    df["hour"] = df["Date and Time"].dt.hour

    # 2A. Reproducing pre-split groupby leakage on random split
    vol = df.groupby(["hour", "channel_id"]).size().reset_index(name="count")
    df_leaked = pd.merge(df, vol, on=["hour", "channel_id"], how="left")
    df_leaked["channel_code"] = LabelEncoder().fit_transform(df_leaked["channel_id"].astype(str))
    df_leaked["veh_code"] = LabelEncoder().fit_transform(df_leaked["veh_class"].astype(str))

    X = df_leaked[["hour", "channel_code", "veh_code"]]
    y = df_leaked["count"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    preds_rand = rf.predict(X_te)
    print(
        f"[Random Split w/ Pre-Split Groupby] Accuracy: {accuracy_score(y_te, preds_rand):.4f} "
        f"| Macro-F1: {f1_score(y_te, preds_rand, average='macro'):.4f}"
    )

    # 2B. Location-Held-Out Split (Hold out ZeroBridge corridor)
    held_out_corridor = "PConvent_ZeroBridge_ANPR_L1"
    train_mark = df_leaked["channel_id"] != held_out_corridor
    rf_loc = RandomForestClassifier(n_estimators=100, random_state=42).fit(
        X[train_mark], y[train_mark]
    )
    preds_loc = rf_loc.predict(X[~train_mark])
    print(
        f"[Location-Held-Out ({held_out_corridor})] Exact-Bucket Accuracy: "
        f"{accuracy_score(y[~train_mark], preds_loc):.4f}"
    )

    # 2C. Strict Time-Based Split (Train hours <= 17, Test hours >= 18)
    time_mask = df_leaked["hour"] <= 17
    rf_time = RandomForestClassifier(n_estimators=100, random_state=42).fit(
        X[time_mask], y[time_mask]
    )
    preds_time = rf_time.predict(X[~time_mask])
    print(
        f"[Chronological Split (Test Hour >= 18:00)] Exact-Bucket Accuracy: "
        f"{accuracy_score(y[~time_mask], preds_time):.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_path", default="STAGE_TIMING Convent.xlsx")
    parser.add_argument("--volume_path", default="p convent.xlsx")
    args = parser.parse_args()
    audit_signal_timing(args.signal_path)
    audit_volume_pipeline(args.volume_path)