"""Load sample documents into BigQuery from raw CSVs.

Transforms support tickets and reviews into a unified documents table,
demonstrating the GCP data pipeline: CSV -> GCS -> BigQuery.
"""

import csv
import json
import uuid
from datetime import datetime, timezone

from google.cloud import bigquery

PROJECT_ID = "project-b95d122b-3728-497b-93f"
DATASET = "nlp_extraction"
TABLE = "documents"

DATA_DIR = "/Users/jessepassmore/Desktop/Programming_Pizazz/nlp_fun/nlp_parsing_gcp/data/raw"


def load_support_tickets(max_rows=200):
    """Load support tickets into the documents schema."""
    rows = []
    path = f"{DATA_DIR}/support_tickets/customer_support_tickets.csv"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            metadata = {
                "ticket_id": row.get("Ticket ID", ""),
                "customer_name": row.get("Customer Name", ""),
                "product": row.get("Product Purchased", ""),
                "ticket_type": row.get("Ticket Type", ""),
                "ticket_subject": row.get("Ticket Subject", ""),
                "priority": row.get("Ticket Priority", ""),
                "status": row.get("Ticket Status", ""),
                "channel": row.get("Ticket Channel", ""),
                "satisfaction": row.get("Customer Satisfaction Rating", ""),
            }
            text = row.get("Ticket Description", "")
            if not text:
                continue
            rows.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "source_type": "support_ticket",
                "metadata": json.dumps(metadata),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
    return rows


def load_reviews(max_rows=200):
    """Load product reviews into the documents schema."""
    rows = []
    path = f"{DATA_DIR}/reviews/archive/1429_1.csv"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            metadata = {
                "product_name": row.get("name", ""),
                "brand": row.get("brand", ""),
                "rating": row.get("reviews.rating", ""),
                "review_title": row.get("reviews.title", ""),
                "username": row.get("reviews.username", ""),
                "categories": row.get("categories", ""),
            }
            text = row.get("reviews.text", "")
            if not text:
                continue
            rows.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "source_type": "review",
                "metadata": json.dumps(metadata),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
    return rows


def main():
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET}.{TABLE}"

    tickets = load_support_tickets(200)
    reviews = load_reviews(200)
    all_rows = tickets + reviews

    print(f"Loaded {len(tickets)} support tickets, {len(reviews)} reviews")
    print(f"Total: {len(all_rows)} documents to insert")

    errors = client.insert_rows_json(table_ref, all_rows)
    if errors:
        print(f"Errors: {errors[:3]}")
    else:
        print(f"Successfully inserted {len(all_rows)} rows into {table_ref}")

    # Verify
    query = f"SELECT source_type, COUNT(*) as cnt FROM `{table_ref}` GROUP BY source_type"
    result = client.query(query).result()
    print("\nVerification:")
    for row in result:
        print(f"  {row.source_type}: {row.cnt} documents")


if __name__ == "__main__":
    main()
