"""Generate realistic fashion retail data with intentional messiness."""
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

random.seed(42)
np.random.seed(42)

BRANDS = [
    "Zara", "H&M", "Uniqlo", "Nike", "Adidas", "Levi's", "Gap",
    "Puma", "Ralph Lauren", "Tommy Hilfiger", "Calvin Klein",
    "Gucci", "Mango", "Forever21", "ASOS",
]

CATEGORIES = ["Tops", "Bottoms", "Dresses", "Outerwear", "Accessories", "Footwear"]

# Intentionally messy — same color written differently
COLORS_MESSY = [
    "Black", "black", "BLACK", "White", "white", "Blue", "blue", "BLUE", "blu",
    "Red", "red", "Green", "green", "Navy", "navy", "Beige", "beige",
    "Pink", "pink", "Gray", "grey", "GREY", "Yellow", "Brown",
    "Neon Green", "Maroon", "Olive", "Coral",
]

SIZES = ["XS", "S", "M", "L", "XL", "XXL", "xs", "Medium", "Med", "LARGE"]

NUM_PRODUCTS = 200
NUM_TRANSACTIONS = 500_000
START_DATE = datetime(2019, 1, 1)
END_DATE = datetime(2024, 12, 31)


def generate():
    engine = create_engine(settings.db_url)

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS transactions"))
        conn.execute(text("DROP TABLE IF EXISTS products"))

        conn.execute(text("""
            CREATE TABLE products (
                product_id   SERIAL PRIMARY KEY,
                product_name VARCHAR(100) NOT NULL,
                brand        VARCHAR(100),
                category     VARCHAR(100),
                color        VARCHAR(50),
                size         VARCHAR(20),
                base_price   NUMERIC(10, 2)
            )
        """))

        conn.execute(text("""
            CREATE TABLE transactions (
                transaction_id SERIAL PRIMARY KEY,
                product_id     INTEGER REFERENCES products(product_id),
                sale_date      DATE,
                quantity       INTEGER,
                unit_price     NUMERIC(10, 2),
                sale_amount    NUMERIC(12, 2),
                purchase_amount NUMERIC(12, 2)
            )
        """))

        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_transactions_sale_date "
            "ON transactions (sale_date)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_transactions_product_id "
            "ON transactions (product_id)"
        ))

    # Products table
    products = []
    for i in range(NUM_PRODUCTS):
        products.append({
            "product_name": f"Product_{i+1}",
            "brand": random.choice(BRANDS),
            "category": random.choice(CATEGORIES),
            "color": random.choice(COLORS_MESSY),
            "size": random.choice(SIZES),
            "base_price": round(random.uniform(10, 500), 2),
        })

    products_df = pd.DataFrame(products)
    products_df.to_sql("products", engine, if_exists="append", index=False, chunksize=10000)

    # Read back with IDs assigned by DB
    with engine.connect() as conn:
        products_with_ids = pd.read_sql("SELECT * FROM products", conn)
    product_list = products_with_ids.to_dict("records")

    # Transactions table
    date_range = (END_DATE - START_DATE).days
    transactions = []

    for i in range(NUM_TRANSACTIONS):
        pid = random.choice(product_list)
        sale_date = START_DATE + timedelta(days=random.randint(0, date_range))

        month = sale_date.month
        seasonal = 1.0
        if pid["category"] == "Outerwear" and month in [11, 12, 1, 2]:
            seasonal = 2.5
        elif pid["category"] == "Outerwear" and month in [6, 7, 8]:
            seasonal = 0.3

        quantity = max(1, int(np.random.lognormal(1, 0.8)))
        unit_price = pid["base_price"] * random.uniform(0.7, 1.3)
        sale_amount = round(quantity * unit_price * seasonal, 2)

        if random.random() < 0.02:
            sale_amount = -sale_amount  # returns
        if random.random() < 0.01:
            sale_date = None  # null dates
        if random.random() < 0.03:
            quantity = None  # null quantity

        transactions.append({
            "product_id": pid["product_id"],
            "sale_date": sale_date,
            "quantity": quantity,
            "unit_price": round(unit_price, 2),
            "sale_amount": sale_amount,
            "purchase_amount": round(sale_amount * random.uniform(0.4, 0.7), 2) if sale_amount > 0 else 0,
        })

    # Inject duplicates (2% of rows)
    dup_count = int(NUM_TRANSACTIONS * 0.02)
    dups = random.sample(transactions, dup_count)
    transactions.extend(dups)
    random.shuffle(transactions)

    trans_df = pd.DataFrame(transactions)
    trans_df.to_sql(
        "transactions", engine, if_exists="append", index=False, chunksize=10000
    )

    total = len(trans_df)
    print(f"Created {len(products_df)} products, {total} transactions")
    print("Intentional issues: ~2% returns, ~1% null dates, ~3% null quantities, ~2% duplicates")
    print(f"Inconsistent values: colors ({len(set(COLORS_MESSY))} variants), sizes ({len(set(SIZES))} variants)")


if __name__ == "__main__":
    generate()
