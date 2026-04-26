"""
producer.py
-----------
Reads creditcard.csv row by row and publishes each transaction as a JSON
message to the Kafka topic "transactions", simulating a real-time data stream.

Run after: docker compose up -d
"""

import csv
import json
import time
import sys
from datetime import datetime

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

# ── Configuration ─────────────────────────────────────────────────────────────
KAFKA_BROKER    = "localhost:9092"   # matches PLAINTEXT_EXTERNAL listener
KAFKA_TOPIC     = "transactions"
CSV_FILE        = "creditcard.csv"
DELAY_SECONDS   = 0.5               # simulate ~2 transactions/second
MAX_RETRIES     = 10                # broker connection retries at startup
RETRY_WAIT      = 3                 # seconds between connection retries


# ── Delivery callback ─────────────────────────────────────────────────────────
def on_send_success(record_metadata):
    """Called by the producer for every successfully delivered message."""
    print(
        f"  [OK] topic={record_metadata.topic} "
        f"partition={record_metadata.partition} "
        f"offset={record_metadata.offset}"
    )


def on_send_error(excp):
    """Called by the producer when a message cannot be delivered."""
    print(f"  [ERROR] Failed to deliver message: {excp}", file=sys.stderr)


# ── Producer factory ──────────────────────────────────────────────────────────
def create_producer(broker: str, retries: int, wait: int) -> KafkaProducer:
    """
    Attempt to connect to Kafka with retries.
    Kafka may take 10-20 s to become ready after docker compose up.
    """
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=broker,
                # Serialize Python dict → UTF-8 JSON bytes
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                # Wait for leader acknowledgement (balance speed vs. durability)
                acks="all",
                # Retry up to 3 times on transient send failures
                retries=3,
                # Batch up to 16 KB before flushing (reduces round-trips)
                batch_size=16_384,
                # Wait up to 10 ms for more messages before sending a batch
                linger_ms=10,
            )
            print(f"[CONNECTED] Kafka broker at {broker}")
            return producer
        except NoBrokersAvailable:
            print(
                f"[WAIT] Broker not ready — attempt {attempt}/{retries}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    print("[FATAL] Could not connect to Kafka. Is docker compose up?", file=sys.stderr)
    sys.exit(1)


# ── Main streaming loop ───────────────────────────────────────────────────────
def stream_csv(csv_path: str, producer: KafkaProducer, topic: str, delay: float):
    """
    Open creditcard.csv and publish one row every `delay` seconds.
    Each message payload is a dict with all CSV columns plus a wall-clock
    timestamp so downstream consumers know when we produced it.
    """
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            total_rows   = 0
            fraud_count  = 0
            start_time   = time.time()

            print(f"\n[START] Streaming '{csv_path}' → topic '{topic}'")
            print(f"        Delay: {delay}s per row  |  Press Ctrl+C to stop\n")

            for row in reader:
                # Cast numeric strings to float (V1-V28, Time, Amount, Class)
                payload = {col: float(val) for col, val in row.items()}

                # Attach a wall-clock timestamp for downstream time-based joins
                payload["event_time"] = datetime.utcnow().isoformat()

                # Derive a synthetic user_id from Time (mod 1000) so we can
                # test per-user feature aggregation without real user IDs
                payload["user_id"] = f"user_{int(payload['Time']) % 1000:04d}"

                # Publish asynchronously; callbacks fire when broker ACKs
                producer.send(topic, value=payload) \
                         .add_callback(on_send_success) \
                         .add_errback(on_send_error)

                total_rows  += 1
                is_fraud     = int(payload["Class"]) == 1
                fraud_count += int(is_fraud)

                # Human-readable progress line
                label    = "FRAUD ⚠" if is_fraud else "legit"
                elapsed  = time.time() - start_time
                rate     = total_rows / elapsed if elapsed > 0 else 0
                print(
                    f"[ROW {total_rows:>6}] user={payload['user_id']}  "
                    f"amount=${payload['Amount']:>8.2f}  "
                    f"class={label:<10}  "
                    f"rate={rate:.1f} rows/s  "
                    f"fraud_total={fraud_count}"
                )

                # Flush every 100 messages so the internal buffer doesn't grow
                # unbounded during long runs
                if total_rows % 100 == 0:
                    producer.flush()

                time.sleep(delay)

        # Final flush to ensure every message is delivered before we exit
        producer.flush()
        elapsed = time.time() - start_time
        print(
            f"\n[DONE] Sent {total_rows:,} rows in {elapsed:.1f}s  "
            f"| fraud rows: {fraud_count} ({fraud_count/total_rows*100:.2f}%)"
        )

    except FileNotFoundError:
        print(
            f"[ERROR] '{csv_path}' not found.\n"
            "Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it in the same directory as producer.py",
            file=sys.stderr,
        )
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n[STOPPED] Keyboard interrupt — flushing remaining messages...")
        producer.flush()
        print("[EXIT] Producer shut down cleanly.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    producer = create_producer(KAFKA_BROKER, MAX_RETRIES, RETRY_WAIT)

    try:
        stream_csv(CSV_FILE, producer, KAFKA_TOPIC, DELAY_SECONDS)
    finally:
        producer.close()
        print("[CLOSED] Kafka producer connection closed.")
