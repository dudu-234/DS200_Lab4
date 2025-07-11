from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.streaming import StreamingContext
from dataloader import DataLoader
from models.utils.feature_extractors import *
import numpy as np

class SparkConfig:
    def __init__(self, app_name, kafka_broker, kafka_topic):
        self.app_name = app_name
        self.kafka_broker = kafka_broker
        self.kafka_topic = kafka_topic

class Trainer:
    def __init__(self, model, mode, spark_config: SparkConfig):
        self.model = model
        self.mode = mode

        self.spark_config = spark_config
        self.spark = SparkSession.builder \
            .appName(self.spark_config.app_name) \
            .master("local[*]") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5,org.apache.kafka:kafka-clients:3.9.0") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        self.sc = self.spark.sparkContext
        self.sql_context = SQLContext(self.sc)

        self.dataloader = DataLoader(
            spark_context=self.sc,
            sql_context=self.sql_context,
            spark_session=self.spark,
            spark_config=self.spark_config,
            mode=self.mode
        )

        self.feature_cols = [
            "vendor_id",
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
            "passenger_count",
            "pickup_date",
            "pickup_hour",
            "pickup_minute",
            "pickup_weekday",
            "is_weekend",
            "pickup_week",
            "pickup_month",
            "pickup_week_hour",
            "haversine_distance",
            "manhattan_distance",
            "direction",
            "center_latitude",
            "center_longitude",
            "pickup_lat_rotated",
            "pickup_lon_rotated",
            "dropoff_lat_rotated",
            "dropoff_lon_rotated",
            "store_and_fwd_flag"
        ]

    def run(self):
        df = self.dataloader.parse_stream()
        
        df = categorise(df)
        df = extract_datetime(df)
        df = extract_distance(df)
        df = extract_direction(df)
        df = extract_center_coordinates(df)
        df = rotate_coordinates(df)

        def train_batch(batch_df, batch_id):
            if batch_df.count() == 0:
                print(f"[Batch {batch_id}] Empty, skipping.")
                return

            print(f"[Batch {batch_id}] Training...")

            pdf = batch_df.select(
                *(self.feature_cols + ["trip_duration"])
            ).toPandas()

            pdf["vendor_id"] = pdf["vendor_id"].astype("category")
            pdf["store_and_fwd_flag"] = pdf["store_and_fwd_flag"].astype("category")

            X = pdf[self.feature_cols]
            y = pdf["trip_duration"]

            self.model.train_batch(X, y)

        def predict_batch(batch_df, batch_id):
            if batch_df.count() == 0:
                print(f"[Batch {batch_id}] Empty, skipping.")
                return

            print(f"[Batch {batch_id}] Predicting...")

            pdf = batch_df.select(*self.feature_cols).dropna().toPandas()
            X = pdf[self.feature_cols]

            preds = self.model.predict_batch(X)
            print(f"[Batch {batch_id}] Prediction sample: {preds[:5]}")

        query = df.writeStream.foreachBatch(
            train_batch if self.mode == "train" else predict_batch
        ).start()

        query.awaitTermination()
