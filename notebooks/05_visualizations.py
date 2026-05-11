import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
sns.set_style("whitegrid")

print("Loading data...")
df = pd.read_csv(f"{OUT}/orders_master_cleaned.csv")
risk = pd.read_csv(f"{OUT}/order_risk_scores.csv")
df = df.merge(risk, on="order_id", how="left")
df["revenue_if_bad"] = df["order_total_value"].fillna(0) * df["return_flag"].fillna(0)

total_orders = len(df)
return_rate = df["return_flag"].mean() * 100
revenue_lost = df["revenue_if_bad"].sum()
print(f"Total orders:    {total_orders:,}")
print(f"Return rate:     {return_rate:.2f}%")
print(f"Revenue at risk: R$ {revenue_lost:,.0f}")

# Chart 1 - KPI summary
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")
kpis = [
    ("TOTAL ORDERS", f"{total_orders:,}"),
    ("RETURN RATE", f"{return_rate:.2f}%"),
    ("REVENUE AT RISK", f"R$ {revenue_lost/1000:,.0f}K"),
]
for i, (label, value) in enumerate(kpis):
    ax.text(0.2 + i * 0.3, 0.7, value, ha="center", fontsize=26, fontweight="bold", color="#1f4e79")
    ax.text(0.2 + i * 0.3, 0.3, label, ha="center", fontsize=12, color="#555")
plt.title("Returns Intelligence AI - Executive Summary", fontsize=18, pad=15)
plt.tight_layout()
plt.savefig(f"{OUT}/01_kpi_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 01_kpi_summary.png")

# Chart 2 - Worst categories
cat = (df.dropna(subset=["product_category_name_english"])
       .groupby("product_category_name_english")
       .agg(n=("order_id", "count"), rate=("return_flag", "mean"),
            revenue_lost=("revenue_if_bad", "sum"))
       .reset_index())
cat = cat[cat["n"] >= 200].copy()
cat["rate"] *= 100
top_bad = cat.sort_values("rate", ascending=False).head(15)

plt.figure(figsize=(11, 7))
sns.barplot(data=top_bad, x="rate", y="product_category_name_english", color="#c00000")
plt.title("Top 15 Categories by Return Rate", fontsize=14)
plt.xlabel("Return Rate (%)")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(f"{OUT}/02_worst_categories.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 02_worst_categories.png")

# Chart 3 - States
state = (df.groupby("customer_state")
         .agg(n=("order_id", "count"), rate=("return_flag", "mean"))
         .reset_index())
state["rate"] *= 100
state = state[state["n"] >= 100].sort_values("rate", ascending=False)

plt.figure(figsize=(13, 6))
sns.barplot(data=state, x="customer_state", y="rate", color="#1f4e79")
plt.title("Return Rate by Brazilian State", fontsize=14)
plt.ylabel("Return Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/03_states.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 03_states.png")

# Chart 4 - Delivery impact
df["delay_bucket"] = pd.cut(df["delivery_delay_days"],
                             bins=[-100, -5, 0, 5, 15, 100],
                             labels=["Very early", "On time", "Late 1-5d", "Late 5-15d", "Very late"])
delay = (df.dropna(subset=["delay_bucket"])
         .groupby("delay_bucket", observed=True)
         .agg(n=("order_id", "count"), rate=("return_flag", "mean"))
         .reset_index())
delay["rate"] *= 100

plt.figure(figsize=(11, 6))
plt.bar(delay["delay_bucket"].astype(str), delay["rate"],
        color=["#a9d18e", "#a9d18e", "#ffc000", "#ed7d31", "#c00000"])
plt.title("Late Delivery Drives Returns", fontsize=14)
plt.ylabel("Return Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUT}/04_delivery_impact.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 04_delivery_impact.png")

# Chart 5 - Risk distribution
risk_dist = df["risk_band"].value_counts().reindex(["Low", "Medium", "High", "Very High"])
plt.figure(figsize=(9, 5))
plt.bar(risk_dist.index.astype(str), risk_dist.values,
        color=["#a9d18e", "#ffc000", "#ed7d31", "#c00000"])
plt.title("AI Model - Order Risk Distribution", fontsize=14)
plt.ylabel("Number of Orders")
for i, v in enumerate(risk_dist.values):
    plt.text(i, v + max(risk_dist.values)*0.01, f"{v:,}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/06_risk_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 06_risk_distribution.png")

# Interactive dashboard - simple Plotly chart
print("\nBuilding interactive dashboard.html...")
fig = px.bar(top_bad, x="rate", y="product_category_name_english",
             orientation="h",
             title="Returns Intelligence AI - Top Risk Categories",
             color="rate", color_continuous_scale="Reds",
             labels={"rate": "Return Rate (%)", "product_category_name_english": "Category"})
fig.update_layout(height=700, template="plotly_white")
fig.write_html(f"{OUT}/dashboard.html")
print("Saved dashboard.html")

# Tableau-ready file
cols = ["order_id", "customer_state", "product_category_name_english",
        "order_total_value", "delivery_delay_days", "primary_payment_type",
        "avg_review_score", "return_flag", "is_canceled", "is_low_review",
        "purchase_year", "purchase_month", "return_risk_score", "risk_band"]
df[[c for c in cols if c in df.columns]].to_csv(f"{OUT}/tableau_ready.csv", index=False)
print("Saved tableau_ready.csv")

print("\nDONE.")