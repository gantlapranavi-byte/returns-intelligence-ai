import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from xgboost import XGBClassifier
import shap

DATA_DIR = "outputs"
os.makedirs(DATA_DIR, exist_ok=True)

print("Loading cleaned data...")
df = pd.read_csv(f"{DATA_DIR}/orders_master_cleaned.csv")

feature_cols_numeric = [
    "n_items",
    "total_price",
    "total_freight",
    "avg_item_price",
    "n_unique_products",
    "n_unique_sellers",
    "order_total_value",
    "payment_total",
    "n_payment_methods",
    "avg_installments",
    "promised_delivery_days",
    "purchase_hour",
    "purchase_dayofweek",
    "purchase_month",
]

feature_cols_categorical = [
    "primary_payment_type",
    "customer_state",
    "product_category_name_english",
]

target = "return_flag"

model_df = df.dropna(subset=[target] + feature_cols_numeric).copy()
print(f"Modeling rows: {len(model_df):,}")

for c in feature_cols_categorical:
    model_df[c] = model_df[c].fillna("unknown").astype(str)

encoders = {}
for c in feature_cols_categorical:
    le = LabelEncoder()
    model_df[c] = le.fit_transform(model_df[c])
    encoders[c] = le

X = model_df[feature_cols_numeric + feature_cols_categorical]
y = model_df[target].astype(int)

print(f"Feature matrix: {X.shape}")
print(f"Class balance: {y.mean()*100:.2f}% positive")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining XGBoost classifier...")
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

print(f"\nROC-AUC:           {auc:.4f}")
print(f"Average Precision: {ap:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

print("Confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()
axes[0].grid(alpha=0.3)

prec, rec, _ = precision_recall_curve(y_test, y_proba)
axes[1].plot(rec, prec, label=f"AP = {ap:.3f}", linewidth=2, color="darkorange")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()
axes[1].grid(alpha=0.3)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=["No Issue", "Bad Outcome"],
            yticklabels=["No Issue", "Bad Outcome"])
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")
axes[2].set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved evaluation plots to {DATA_DIR}/model_evaluation.png")

importances = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nTop 10 features:")
print(importances.head(10).to_string(index=False))
importances.to_csv(f"{DATA_DIR}/feature_importance.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importances.head(15), x="importance", y="feature", color="steelblue")
plt.title("Top 15 Features - XGBoost")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nComputing SHAP values (this is the explainable-AI part)...")

sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, sample, show=False, max_display=15)
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved SHAP summary to {DATA_DIR}/shap_summary.png")

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, sample, plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

joblib.dump({
    "model": model,
    "encoders": encoders,
    "feature_cols": feature_cols_numeric + feature_cols_categorical,
    "metrics": {"auc": auc, "average_precision": ap},
}, f"{DATA_DIR}/returns_model.pkl")
print(f"\nSaved model to {DATA_DIR}/returns_model.pkl")

print("\nScoring all orders for the dashboard...")
all_proba = model.predict_proba(X)[:, 1]
scored = model_df[["order_id"]].copy()
scored["return_risk_score"] = all_proba
scored["risk_band"] = pd.cut(
    all_proba,
    bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
    labels=["Low", "Medium", "High", "Very High"],
)
scored.to_csv(f"{DATA_DIR}/order_risk_scores.csv", index=False)
print(f"Saved risk scores to {DATA_DIR}/order_risk_scores.csv")

print("\nDONE.")