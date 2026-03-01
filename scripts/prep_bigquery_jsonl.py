"""Prepare JSONL file for BigQuery loading from raw CSVs.

Creates a newline-delimited JSON file that bq load can ingest directly.
"""

import csv
import json
import uuid
from datetime import datetime, timezone

DATA_DIR = "/Users/jessepassmore/Desktop/Programming_Pizazz/nlp_fun/nlp_parsing_gcp/data/raw"
OUTPUT = "/Users/jessepassmore/Desktop/Programming_Pizazz/nlp_fun/nlp_parsing_gcp/data/documents.jsonl"


def main():
    rows = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Support tickets
    path = f"{DATA_DIR}/support_tickets/customer_support_tickets.csv"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 200:
                break
            text = row.get("Ticket Description", "")
            if not text:
                continue
            metadata = {
                "ticket_id": row.get("Ticket ID", ""),
                "product": row.get("Product Purchased", ""),
                "ticket_type": row.get("Ticket Type", ""),
                "subject": row.get("Ticket Subject", ""),
                "priority": row.get("Ticket Priority", ""),
            }
            rows.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "source_type": "support_ticket",
                "metadata": json.dumps(metadata),
                "created_at": now,
            })

    # Reviews
    path = f"{DATA_DIR}/reviews/archive/1429_1.csv"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 200:
                break
            text = row.get("reviews.text", "")
            if not text:
                continue
            metadata = {
                "product": row.get("name", ""),
                "brand": row.get("brand", ""),
                "rating": row.get("reviews.rating", ""),
                "title": row.get("reviews.title", ""),
            }
            rows.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "source_type": "review",
                "metadata": json.dumps(metadata),
                "created_at": now,
            })

    with open(OUTPUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
