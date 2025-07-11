from pyspark.sql import functions as F
from pyspark.sql.functions import col, abs, radians, sin, cos, sqrt, atan2, degrees, lit, udf
from pyspark.sql.types import DoubleType
from pyspark.sql import DataFrame
import math

def categorise(df: DataFrame) -> DataFrame:
    df = df.withColumn("store_and_fwd_flag", F.when(col("store_and_fwd_flag") == "N", 0).otherwise(1))
    return df

def extract_datetime(df: DataFrame) -> DataFrame:
    df = df.withColumn("pickup_datetime", F.to_timestamp("pickup_datetime"))

    if "dropoff_datetime" in df.columns:
        df = df.withColumn("dropoff_datetime", F.to_timestamp("dropoff_datetime"))

    df = df.withColumn("pickup_date", F.dayofmonth("pickup_datetime"))
    df = df.withColumn("pickup_hour", F.hour("pickup_datetime"))
    df = df.withColumn("pickup_minute", F.minute("pickup_datetime"))
    df = df.withColumn("pickup_weekday", F.dayofweek("pickup_datetime"))
    df = df.withColumn("is_weekend", F.when(col("pickup_weekday").isin([1, 7]), 1).otherwise(0))
    df = df.withColumn("pickup_month", F.month("pickup_datetime"))
    df = df.withColumn("pickup_week", F.weekofyear("pickup_datetime"))
    df = df.withColumn("pickup_week_hour", col("pickup_weekday") * 24 + col("pickup_hour"))
    
    return df

def extract_distance(df: DataFrame) -> DataFrame:
    R = 6371.0  # Earth radius in km
    lat1 = radians(col("pickup_latitude"))
    lon1 = radians(col("pickup_longitude"))
    lat2 = radians(col("dropoff_latitude"))
    lon2 = radians(col("dropoff_longitude"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    d = R * c
    df = df.withColumn("haversine_distance", d)
    
    df = df.withColumn("manhattan_distance",
                 abs(col("pickup_latitude") - col("dropoff_latitude")) +
                 abs(col("pickup_longitude") - col("dropoff_longitude")))
    
    return df
    
def extract_direction(df: DataFrame) -> DataFrame:
    lat1 = radians(col("pickup_latitude"))
    lon1 = radians(col("pickup_longitude"))
    lat2 = radians(col("dropoff_latitude"))
    lon2 = radians(col("dropoff_longitude"))

    dlon = lon2 - lon1
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)

    bearing = (degrees(atan2(y, x)) + lit(360.0)) % lit(360.0)

    return df.withColumn("direction", bearing)

def extract_center_coordinates(df: DataFrame) -> DataFrame:
    df = df.withColumn("center_latitude",
                       (col("pickup_latitude") + col("dropoff_latitude")) / lit(2.0))
    df = df.withColumn("center_longitude",
                       (col("pickup_longitude") + col("dropoff_longitude")) / lit(2.0))
    
    return df

def rotate_coordinates(df: DataFrame) -> DataFrame:
    # Manhattan grid is offset by 29 degrees compared to the true north
    # Roration helps model understand the route better
    THETA = -29

    theta_rad = lit(math.radians(THETA))
    cos_theta = cos(theta_rad)
    sin_theta = sin(theta_rad)

    # Coordinate of NYC center
    mean_lon = -73.9712
    mean_lat = 40.7831

    pickup_x = col("pickup_latitude") - lit(mean_lat)
    pickup_y = col("pickup_longitude") - lit(mean_lon)
    pickup_rot0 = pickup_x * lit(cos_theta) - pickup_y * lit(sin_theta)
    pickup_rot1 = pickup_x * lit(sin_theta) + pickup_y * lit(cos_theta)

    dropoff_x = col("dropoff_latitude") - lit(mean_lat)
    dropoff_y = col("dropoff_longitude") - lit(mean_lon)
    dropoff_rot0 = dropoff_x * lit(cos_theta) - dropoff_y * lit(sin_theta)
    dropoff_rot1 = dropoff_x * lit(sin_theta) + dropoff_y * lit(cos_theta)

    return df.withColumn("pickup_lat_rotated", pickup_rot0)\
           .withColumn("pickup_lon_rotated", pickup_rot1)\
           .withColumn("dropoff_lat_rotated", dropoff_rot0)\
           .withColumn("dropoff_lon_rotated", dropoff_rot1)
