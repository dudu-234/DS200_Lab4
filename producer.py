import pandas as pd
import json
import argparse
import time
from kafka import KafkaProducer

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True, help="Path to train.csv or test.csv")
parser.add_argument("--batch-size", type=int, default=1024, help="Batch size per message")
parser.add_argument("--sleep", type=float, default=2, help="Sleep time between batches (seconds)")
parser.add_argument("--mode", type=str, choices=["train", "predict"], default="train", help="train or predict mode")
parser.add_argument("--bootstrap-servers", type=str, default="localhost:9092", help="Kafka bootstrap servers")
parser.add_argument("--topic", type=str, default="nyc-taxi-trip-duration", help="Kafka topic")

args = parser.parse_args()

producer = KafkaProducer(
    bootstrap_servers=args.bootstrap_servers,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

df = pd.read_csv(args.file)

feature_columns = [
    "vendor_id", "pickup_datetime", "passenger_count",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "store_and_fwd_flag"
]
if args.mode == "train":
    feature_columns.extend(["dropoff_datetime", "trip_duration"])  # include label in train mode

total_rows = len(df)
batch_size = args.batch_size

print(f"Starting Kafka Producer in {args.mode.upper()} mode, total {total_rows} rows")

for idx in range(0, total_rows, batch_size):
    batch_df = df.iloc[idx:idx+batch_size]
    for row_idx, row in batch_df.iterrows():
        payload = {col: row[col] for col in feature_columns}

    try:
        producer.send(args.topic, payload)
        print(f"Sent batch {idx // batch_size + 1}, rows {idx} to {idx + len(batch_df) - 1}")
    except Exception as e:
        print(f"Error sending batch: {e}")

    time.sleep(args.sleep)

print("Finished sending all batches.")
producer.flush()
producer.close()
