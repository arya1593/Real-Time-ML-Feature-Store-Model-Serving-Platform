"""
spark_streaming.py
------------------
Structured Streaming job that:
  1. Reads JSON transaction messages from Kafka topic "transactions"
  2. Computes per-user, 5-minute windowed features (count, avg_amount, max_amount)
  3. Writes live features to Redis  (real-time feature store)
  4. Writes raw transactions to Parquet (historical / Hive-compatible store)

Run AFTER producer.py is streaming data.

Prerequisites (one-time setup):
  pip install pyspark redis
  Java 11 must be installed and JAVA_HOME must be set.

Run command:
  python spark_streaming.py
  (The Kafka connector JAR is fetched automatically via spark.jars.packages)
"""

import json
import os
import sys
from datetime import datetime

import redis
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ── Configuration ─────────────────────────────────────────────────────────────
KAFKA_BROKER        = "localhost:9092"
KAFKA_TOPIC         = "transactions"
REDIS_HOST          = "localhost"
REDIS_PORT          = 6379
REDIS_TTL_SECONDS   = 3600          # expire feature keys after 1 hour
WINDOW_DURATION     = "5 minutes"   # aggregation window size
SLIDE_DURATION      = "1 minute"    # how often the window updates (sliding)
WATERMARK_DELAY     = "2 minutes"   # tolerate late messages up to 2 min
CHECKPOINT_DIR      = "./checkpoints"
PARQUET_OUTPUT_DIR  = "./hive_warehouse/transactions"

# Kafka connector package — Spark downloads this automatically on first run
KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"


# ── JSON Schema (mirrors creditcard.csv + enrichment fields from producer.py) ─
# Defining an explicit schema is faster than schema inference and avoids
# Spark making an extra Kafka round-trip.
TRANSACTION_SCHEMA = StructType([
    StructField("Time",       DoubleType(),   True),
    StructField("V1",         DoubleType(),   True),
    StructField("V2",         DoubleType(),   True),
    StructField("V3",         DoubleType(),   True),
    StructField("V4",         DoubleType(),   True),
    StructField("V5",         DoubleType(),   True),
    StructField("V6",         DoubleType(),   True),
    StructField("V7",         DoubleType(),   True),
    StructField("V8",         DoubleType(),   True),
    StructField("V9",         DoubleType(),   True),
    StructField("V10",        DoubleType(),   True),
    StructField("V11",        DoubleType(),   True),
    StructField("V12",        DoubleType(),   True),
    StructField("V13",        DoubleType(),   True),
    StructField("V14",        DoubleType(),   True),
    StructField("V15",        DoubleType(),   True),
    StructField("V16",        DoubleType(),   True),
    StructField("V17",        DoubleType(),   True),
    StructField("V18",        DoubleType(),   True),
    StructField("V19",        DoubleType(),   True),
    StructField("V20",        DoubleType(),   True),
    StructField("V21",        DoubleType(),   True),
    StructField("V22",        DoubleType(),   True),
    StructField("V23",        DoubleType(),   True),
    StructField("V24",        DoubleType(),   True),
    StructField("V25",        DoubleType(),   True),
    StructField("V26",        DoubleType(),   True),
    StructField("V27",        DoubleType(),   True),
    StructField("V28",        DoubleType(),   True),
    StructField("Amount",     DoubleType(),   True),
    StructField("Class",      DoubleType(),   True),
    StructField("event_time", StringType(),   True),  # ISO-8601 from producer
    StructField("user_id",    StringType(),   True),
])


# ── SparkSession ──────────────────────────────────────────────────────────────
def build_spark_session() -> SparkSession:
    """
    Create a local SparkSession with the Kafka connector.
    'local[*]' uses all available CPU cores.
    The Kafka JAR is resolved from Maven on first run (~30 s download).
    """
    spark = (
        SparkSession.builder
        .appName("FraudFeatureStore")
        .master("local[*]")
        # Auto-download the Kafka connector JAR
        .config("spark.jars.packages", KAFKA_PACKAGE)
        # Keep shuffle partitions low for a local single-node run
        .config("spark.sql.shuffle.partitions", "4")
        # Allow legacy time parsing so ISO-8601 strings parse correctly
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        # Point Hive warehouse to our local directory
        .config("spark.sql.warehouse.dir", os.path.abspath(PARQUET_OUTPUT_DIR))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")  # suppress INFO noise
    print("[SPARK] Session created — version:", spark.version)
    return spark


# ── Kafka source ──────────────────────────────────────────────────────────────
def read_kafka_stream(spark: SparkSession):
    """
    Subscribe to the Kafka topic and return a raw streaming DataFrame.
    Each row has: key (bytes), value (bytes), topic, partition, offset, timestamp.
    'startingOffsets=latest' means we only process messages arriving after startup.
    """
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        # Read up to 1000 records per micro-batch to avoid overwhelming Redis
        .option("maxOffsetsPerTrigger", 1000)
        .load()
    )


# ── Parse JSON payload ────────────────────────────────────────────────────────
def parse_transactions(raw_df):
    """
    Kafka delivers messages as binary. Cast value → string → parse as JSON.
    Also cast event_time string → TimestampType for windowing.
    """
    parsed = (
        raw_df
        # Kafka value column is BinaryType; decode to UTF-8 string
        .select(F.col("value").cast("string").alias("json_str"))
        # Expand all fields from the JSON string using our predefined schema
        .select(F.from_json(F.col("json_str"), TRANSACTION_SCHEMA).alias("data"))
        .select("data.*")
        # Parse the ISO-8601 event_time string into a proper Spark timestamp
        .withColumn(
            "event_timestamp",
            F.to_timestamp(F.col("event_time"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS")
        )
    )
    return parsed


# ── Windowed feature aggregation ──────────────────────────────────────────────
def compute_windowed_features(parsed_df):
    """
    Sliding 5-minute window, sliding every 1 minute, per user_id.
    Features computed:
      - txn_count  : number of transactions in the window
      - avg_amount : mean transaction amount
      - max_amount : largest single transaction amount

    Watermarking tells Spark it can safely discard state for events older
    than WATERMARK_DELAY, preventing unbounded memory growth.
    """
    return (
        parsed_df
        .withWatermark("event_timestamp", WATERMARK_DELAY)
        .groupBy(
            F.col("user_id"),
            F.window(F.col("event_timestamp"), WINDOW_DURATION, SLIDE_DURATION)
        )
        .agg(
            F.count("*")             .alias("txn_count"),
            F.avg("Amount")          .alias("avg_amount"),
            F.max("Amount")          .alias("max_amount"),
        )
        # Flatten the window struct into readable columns
        .withColumn("window_start", F.col("window.start").cast("string"))
        .withColumn("window_end",   F.col("window.end").cast("string"))
        .drop("window")
    )


# ── Redis sink ────────────────────────────────────────────────────────────────
def write_features_to_redis(batch_df, batch_id: int):
    """
    foreachBatch sink: called once per micro-batch with a static DataFrame.
    Writes each user's latest windowed features to Redis as a JSON string.

    Key format:  features:<user_id>
    Value:       JSON with txn_count, avg_amount, max_amount, window timestamps
    TTL:         REDIS_TTL_SECONDS (1 hour by default)

    A new Redis connection is created per batch to avoid serialization issues
    when Spark distributes tasks across executor threads.
    """
    if batch_df.rdd.isEmpty():
        print(f"  [REDIS] Batch {batch_id} — empty, skipping.")
        return

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()  # fail fast if Redis is unreachable

        rows = batch_df.collect()   # bring batch to driver (small — aggregated)
        written = 0

        for row in rows:
            key = f"features:{row['user_id']}"
            value = json.dumps({
                "user_id":      row["user_id"],
                "txn_count":    int(row["txn_count"]),
                "avg_amount":   round(float(row["avg_amount"]), 4),
                "max_amount":   round(float(row["max_amount"]), 4),
                "window_start": row["window_start"],
                "window_end":   row["window_end"],
                "updated_at":   datetime.utcnow().isoformat(),
            })
            r.setex(key, REDIS_TTL_SECONDS, value)
            written += 1

        print(
            f"  [REDIS] Batch {batch_id} — wrote {written} feature records "
            f"| sample key: features:{rows[0]['user_id']}"
        )

    except redis.exceptions.ConnectionError as e:
        print(f"  [REDIS ERROR] Batch {batch_id} — could not connect: {e}")


# ── Hive / Parquet sink ───────────────────────────────────────────────────────
def write_raw_to_parquet(batch_df, batch_id: int):
    """
    foreachBatch sink: appends raw transaction rows to Parquet files
    partitioned by date. In a real Hive setup, these Parquet files would
    sit directly in a Hive external table's warehouse directory.

    Partition column 'txn_date' is derived from event_timestamp so Hive
    partition pruning works correctly (e.g. WHERE txn_date='2024-01-15').
    """
    if batch_df.rdd.isEmpty():
        return

    try:
        enriched = batch_df.withColumn(
            "txn_date",
            F.to_date(F.col("event_timestamp"))
        )
        (
            enriched
            .write
            .mode("append")
            .partitionBy("txn_date")   # one sub-directory per day
            .parquet(PARQUET_OUTPUT_DIR)
        )
        count = batch_df.count()
        print(f"  [HIVE]  Batch {batch_id} — appended {count} rows to Parquet")

    except Exception as e:
        print(f"  [HIVE ERROR] Batch {batch_id} — {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Fraud Feature Store — Spark Structured Streaming")
    print("=" * 60)

    # Verify Redis is reachable before starting Spark
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        r.ping()
        print(f"[OK] Redis reachable at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        print(f"[FATAL] Cannot reach Redis: {e}\nIs docker compose up?")
        sys.exit(1)

    spark = build_spark_session()

    # ── Build the streaming pipeline ──────────────────────────────────────────
    raw_df     = read_kafka_stream(spark)
    parsed_df  = parse_transactions(raw_df)
    feature_df = compute_windowed_features(parsed_df)

    # ── Query 1: windowed features → Redis ────────────────────────────────────
    # trigger every 30 seconds so features are reasonably fresh
    redis_query = (
        feature_df
        .writeStream
        .outputMode("update")           # only emit updated windows
        .foreachBatch(write_features_to_redis)
        .trigger(processingTime="30 seconds")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/redis")
        .start()
    )
    print("[STREAM] Redis feature writer started.")

    # ── Query 2: raw transactions → Parquet / Hive ────────────────────────────
    # trigger every 60 seconds — historical store doesn't need to be real-time
    hive_query = (
        parsed_df
        .writeStream
        .outputMode("append")
        .foreachBatch(write_raw_to_parquet)
        .trigger(processingTime="60 seconds")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/hive")
        .start()
    )
    print("[STREAM] Hive/Parquet writer started.")

    # ── Console sink for debugging (shows feature aggregations in terminal) ───
    console_query = (
        feature_df
        .writeStream
        .outputMode("update")
        .format("console")
        .option("truncate", False)
        .option("numRows", 10)
        .trigger(processingTime="30 seconds")
        .start()
    )
    print("[STREAM] Console debug sink started.")
    print("\n[RUNNING] Streaming... Press Ctrl+C to stop.\n")

    try:
        # Block until all queries terminate or an error occurs
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("\n[STOPPING] Shutting down streaming queries...")
        redis_query.stop()
        hive_query.stop()
        console_query.stop()
        print("[DONE] All streams stopped.")
    finally:
        spark.stop()
        print("[CLOSED] SparkSession closed.")


if __name__ == "__main__":
    main()
