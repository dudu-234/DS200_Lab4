from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType, TimestampType

class DataLoader:
    def __init__(self, spark_context, sql_context, spark_session, spark_config, mode):
        self.sc = spark_context
        self.sql_context = sql_context
        self.spark = spark_session
        self.spark_config = spark_config
        self.mode = mode

        self.kafka_broker = self.spark_config.kafka_broker
        self.kafka_topic = self.spark_config.kafka_topic

        if self.mode == "train":
            self.schema = StructType([
                StructField("vendor_id", StringType(), True),
                StructField("pickup_datetime", TimestampType(), True),
                StructField("dropoff_datetime", TimestampType(), True),
                StructField("passenger_count", IntegerType(), True),
                StructField("pickup_longitude", DoubleType(), True),
                StructField("pickup_latitude", DoubleType(), True),
                StructField("dropoff_longitude", DoubleType(), True),
                StructField("dropoff_latitude", DoubleType(), True),
                StructField("store_and_fwd_flag", StringType(), True),
                StructField("trip_duration", IntegerType(), True)
            ])
        elif self.mode == "predict":
            self.schema = StructType([
                StructField("vendor_id", StringType(), True),
                StructField("pickup_datetime", TimestampType(), True),
                StructField("passenger_count", IntegerType(), True),
                StructField("pickup_longitude", DoubleType(), True),
                StructField("pickup_latitude", DoubleType(), True),
                StructField("dropoff_longitude", DoubleType(), True),
                StructField("dropoff_latitude", DoubleType(), True),
                StructField("store_and_fwd_flag", StringType(), True)
            ])

    def parse_stream(self):
        kafka_stream = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_broker) \
            .option("subscribe", self.kafka_topic) \
            .option("startingOffsets", "earliest") \
            .load()
        
        json_df = kafka_stream.selectExpr("CAST(value AS STRING) as json_str")
        json_df = json_df.withColumn("json_data", from_json(col("json_str"), self.schema))

        for field in self.schema.fields:
            json_df = json_df.withColumn(field.name, col("json_data").getItem(field.name))

        json_df = json_df.drop("json_str", "json_data")

        return json_df