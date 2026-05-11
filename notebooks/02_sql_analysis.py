import os
import sqlite3
import pandas as pd

DATA_DIR = "outputs"
DB_PATH = f"{DATA_DIR}/returns_intelligence.sqlite"

print("Loading cleaned data into SQLite...")
df = pd.read_csv(f"{DATA_DIR}/orders_master_cleaned.csv")

conn = sqlite3.connect(DB_PATH)
df.to_sql("orders_master", conn, if_exists="replace", index=False)
print(f"Loaded {len(df):,} rows into 'orders_master' table.")

queries = {
    "q1_overall_metrics": """
        SELECT
            COUNT(*) AS total_orders,
            SUM(return_flag) AS bad_outcome_orders,
            ROUND(AVG(return_flag) * 100, 2) AS bad_outcome_rate_pct,
            ROUND(SUM(CASE WHEN return_flag = 1 THEN order_total_value ELSE 0 END), 2) AS revenue_at_risk,
            ROUND(SUM(order_total_value), 2) AS total_revenue
        FROM orders_master;
    """,
    "q2_worst_categories": """
        SELECT
            product_category_name_english AS category,
            COUNT(*) AS n_orders,
            SUM(return_flag) AS n_bad_outcomes,
            ROUND(AVG(return_flag) * 100, 2) AS return_rate_pct,
            ROUND(AVG(order_total_value), 2) AS avg_order_value,
            ROUND(SUM(CASE WHEN return_flag = 1 THEN order_total_value ELSE 0 END), 2) AS revenue_lost
        FROM orders_master
        WHERE product_category_name_english IS NOT NULL
        GROUP BY product_category_name_english
        HAVING COUNT(*) >= 200
        ORDER BY return_rate_pct DESC
        LIMIT 15;
    """,
    "q3_worst_states": """
        SELECT
            customer_state AS state,
            COUNT(*) AS n_orders,
            ROUND(AVG(return_flag) * 100, 2) AS return_rate_pct,
            ROUND(AVG(delivery_delay_days), 2) AS avg_delivery_delay,
            ROUND(SUM(CASE WHEN return_flag = 1 THEN order_total_value ELSE 0 END), 2) AS revenue_lost
        FROM orders_master
        GROUP BY customer_state
        HAVING COUNT(*) >= 100
        ORDER BY return_rate_pct DESC;
    """,
    "q4_delivery_delay_impact": """
        SELECT
            CASE
                WHEN delivery_delay_days IS NULL THEN 'Not delivered'
                WHEN delivery_delay_days <= -5 THEN 'Very early (5+ days)'
                WHEN delivery_delay_days < 0  THEN 'Early'
                WHEN delivery_delay_days = 0  THEN 'On time'
                WHEN delivery_delay_days <= 5 THEN 'Late (1-5 days)'
                ELSE 'Very late (5+ days)'
            END AS delivery_bucket,
            COUNT(*) AS n_orders,
            ROUND(AVG(return_flag) * 100, 2) AS return_rate_pct
        FROM orders_master
        GROUP BY delivery_bucket
        ORDER BY return_rate_pct DESC;
    """,
    "q5_monthly_trend": """
        SELECT
            purchase_year,
            purchase_month,
            COUNT(*) AS n_orders,
            SUM(return_flag) AS n_bad_outcomes,
            ROUND(AVG(return_flag) * 100, 2) AS return_rate_pct,
            ROUND(SUM(CASE WHEN return_flag = 1 THEN order_total_value ELSE 0 END), 2) AS revenue_lost
        FROM orders_master
        WHERE purchase_year IS NOT NULL
        GROUP BY purchase_year, purchase_month
        ORDER BY purchase_year, purchase_month;
    """,
    "q6_payment_method_impact": """
        SELECT
            primary_payment_type,
            COUNT(*) AS n_orders,
            ROUND(AVG(return_flag) * 100, 2) AS return_rate_pct,
            ROUND(AVG(avg_installments), 2) AS avg_installments
        FROM orders_master
        WHERE primary_payment_type IS NOT NULL
        GROUP BY primary_payment_type
        ORDER BY return_rate_pct DESC;
    """,
    "q7_value_vs_returns": """
        SELECT
            CASE
                WHEN order_total_value < 50  THEN '01_Under_50'
                WHEN order_total_value < 100 THEN '02_50_to_100'
                WHEN order_total_value < 200 THEN '03_100_to_200'
                WHEN order_total_value < 500 THEN '04_200_to_500'
                ELSE '05_Over_500'
            END AS value_bucket,
            COUNT(*) AS n_orders,
            ROUND(AVG(return_flag) * 100, 2) AS return_rate_pct
        FROM orders_master
        WHERE order_total_value IS NOT NULL
        GROUP BY value_bucket
        ORDER BY value_bucket;
    """,
}

print("\nRunning business queries...\n")
for name, sql in queries.items():
    print(f"--- {name} ---")
    result = pd.read_sql_query(sql, conn)
    print(result.to_string(index=False))
    out_file = f"{DATA_DIR}/{name}.csv"
    result.to_csv(out_file, index=False)
    print(f"Saved to {out_file}\n")

conn.close()
print("Done. SQLite database saved to:", DB_PATH)