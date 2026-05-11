import os
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading raw CSV files...")
orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv")
order_items = pd.read_csv(f"{DATA_DIR}/olist_order_items_dataset.csv")
payments = pd.read_csv(f"{DATA_DIR}/olist_order_payments_dataset.csv")
reviews = pd.read_csv(f"{DATA_DIR}/olist_order_reviews_dataset.csv")
customers = pd.read_csv(f"{DATA_DIR}/olist_customers_dataset.csv")
products = pd.read_csv(f"{DATA_DIR}/olist_products_dataset.csv")
sellers = pd.read_csv(f"{DATA_DIR}/olist_sellers_dataset.csv")
category_translation = pd.read_csv(f"{DATA_DIR}/product_category_name_translation.csv")

print(f"  orders:         {orders.shape}")
print(f"  order_items:    {order_items.shape}")
print(f"  payments:       {payments.shape}")
print(f"  reviews:        {reviews.shape}")
print(f"  customers:      {customers.shape}")
print(f"  products:       {products.shape}")
print(f"  sellers:        {sellers.shape}")

print("\nConverting date columns...")
date_cols = ["order_purchase_timestamp", "order_approved_at",
             "order_delivered_carrier_date", "order_delivered_customer_date",
             "order_estimated_delivery_date"]
for c in date_cols:
    orders[c] = pd.to_datetime(orders[c], errors="coerce")

print("Translating categories to English...")
products = products.merge(category_translation, on="product_category_name", how="left")
products["product_category_name_english"] = products["product_category_name_english"].fillna("unknown")

print("Aggregating items to order level...")
order_items_agg = (order_items.groupby("order_id").agg(
    n_items=("order_item_id", "count"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    avg_item_price=("price", "mean"),
    n_unique_products=("product_id", "nunique"),
    n_unique_sellers=("seller_id", "nunique")).reset_index())
order_items_agg["order_total_value"] = order_items_agg["total_price"] + order_items_agg["total_freight"]

payments_agg = (payments.groupby("order_id").agg(
    payment_total=("payment_value", "sum"),
    n_payment_methods=("payment_type", "nunique"),
    avg_installments=("payment_installments", "mean"),
    primary_payment_type=("payment_type", lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")).reset_index())

reviews_agg = (reviews.groupby("order_id").agg(
    avg_review_score=("review_score", "mean"),
    review_message=("review_comment_message", "first")).reset_index())

print("Merging all tables into a master DataFrame...")
df = (orders
      .merge(customers, on="customer_id", how="left")
      .merge(order_items_agg, on="order_id", how="left")
      .merge(payments_agg, on="order_id", how="left")
      .merge(reviews_agg, on="order_id", how="left"))

first_product = (order_items.sort_values("order_item_id")
                 .drop_duplicates("order_id", keep="first")[["order_id", "product_id", "seller_id"]])
first_product = first_product.merge(products[["product_id", "product_category_name_english"]],
                                     on="product_id", how="left")
df = df.merge(first_product[["order_id", "product_category_name_english"]], on="order_id", how="left")

print("Engineering features...")
df["delivery_delay_days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
df["actual_delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
df["promised_delivery_days"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days
df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
df["purchase_month"] = df["order_purchase_timestamp"].dt.month
df["purchase_year"] = df["order_purchase_timestamp"].dt.year

df["is_canceled"] = (df["order_status"] == "canceled").astype(int)
df["is_low_review"] = (df["avg_review_score"] <= 2).astype(int)
df["return_flag"] = ((df["is_canceled"] == 1) | (df["is_low_review"] == 1)).astype(int)

print(f"\nTotal orders:         {len(df):,}")
print(f"Canceled orders:      {df['is_canceled'].sum():,}  ({df['is_canceled'].mean()*100:.2f}%)")
print(f"Low-review orders:    {df['is_low_review'].sum():,}  ({df['is_low_review'].mean()*100:.2f}%)")
print(f"Return-flagged total: {df['return_flag'].sum():,}  ({df['return_flag'].mean()*100:.2f}%)")

out_path = f"{OUTPUT_DIR}/orders_master_cleaned.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved cleaned dataset to {out_path}")
print(f"Shape: {df.shape}")
