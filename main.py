import argparse
from trainer import Trainer, SparkConfig
from models.xgboost import XGBWarmStart
from models.lightgbm import LGBMWarmStart

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["xgb", "lgbm"], required=True, help="Model to use")
parser.add_argument("--model-dir", type=str, default="checkpoint", help="Path to the directory of pre-trained model")
parser.add_argument("--mode", type=str, choices=["train", "predict"], default="train", help="train or predict mode")
parser.add_argument("--kafka-broker", type=str, default="localhost:9092", help="Kafka bootstrap servers")
parser.add_argument("--kafka-topic", type=str, default="nyc-taxi-trip-duration", help="Kafka topic")
parser.add_argument("--spark-app-name", type=str, default="nyc-taxi-trip-duration", help="Spark application name")
args = parser.parse_args()

if __name__ == "__main__":
    spark_config = SparkConfig(
        app_name=args.spark_app_name,
        kafka_broker=args.kafka_broker,
        kafka_topic=args.kafka_topic
    )

    if args.model == "xgb":
        model = XGBWarmStart(
            model_path=f"{args.model_dir}/xgb_model.pkl", 
            enable_categorical=True,
            device="cuda"
        )
    elif args.model == "lgbm":
        model = LGBMWarmStart(
            model_path=f"{args.model_dir}/lgbm_model.pkl", 
            device="gpu"
        )

    trainer = Trainer(model=model, mode=args.mode, spark_config=spark_config)
    trainer.run()
