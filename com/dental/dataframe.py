from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("Melbourne") \
    .master("local[*]") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()
# Add LEGACY config for good datetime parsing because we use spark 3+

pedestrian_count_path = "/home/ahmad/Documents/Free/Spark-SA/Pedestrian_Counting_System_-_Monthly__counts_per_hour.csv"
sensor_location_path = "/home/ahmad/Documents/Free/Spark-SA/Pedestrian_Counting_System_-_Sensor_Locations.csv"
#Data Loading
#Question 1
pedestrian_count_schema = StructType([
    StructField("ID", IntegerType(), True),
    StructField("Date_Time", StringType(), True),
    StructField("Year", IntegerType(), True),
    StructField("Month", StringType(), True),
    StructField("Mdate", IntegerType(), True),
    StructField("Day", StringType(), True),
    StructField("Time", IntegerType(), True),
    StructField("Sensor_ID", IntegerType(), True),
    StructField("Sensor_Name", StringType(), True),
    StructField("Hourly_Counts", IntegerType(), True)])

sensor_location_schema = StructType([
    StructField("sensor_id", IntegerType(), True),
    StructField("sensor_description", StringType(), True),
    StructField("sensor_name", StringType(), True),
    StructField("installation_date", StringType(), True),
    StructField("status", StringType(), True),
    StructField("note", StringType(), True),
    StructField("direction_1", StringType(), True),
    StructField("direction_2", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("location", StringType(), True)])

#Question 2
pedestrian_count_df = spark.read.format("csv").schema(pedestrian_count_schema) \
    .load(pedestrian_count_path)
sensor_location_df = spark.read.format("csv").option("header", "true").schema(sensor_location_schema) \
    .load(sensor_location_path)
pedestrian_count_df.printSchema()
sensor_location_df.printSchema()

#Question 3a
pedestrian_count_df \
    .select("Date_Time") \
    .where((pedestrian_count_df.ID == 2853222) | (pedestrian_count_df.ID == 2853223)) \
    .show()

pedestrian_count_df = pedestrian_count_df. \
    withColumn("Date_Time", unix_timestamp("Date_Time", 'MM/dd/yyyy HH:mm:ss aa').cast(TimestampType()))

pedestrian_count_df \
    .select("Date_Time") \
    .where((pedestrian_count_df.ID == 2853222) | (pedestrian_count_df.ID == 2853223)) \
    .show()
#Question 3b
sensor_location_df = sensor_location_df \
    .withColumn("installation_date", unix_timestamp("installation_date", 'MM/dd/yyyy').cast(TimestampType()))
#Question 3c
sensor_location_df = sensor_location_df.withColumn("location", regexp_extract("location", "-?[0-9]*\.*[0-9]*,\s*-?[0-9]*\.*[0-9]*", 0)) \
    .withColumn("location", split("location", ',\s*').cast(ArrayType(DoubleType())))
#Question 3d
sensor_location_df.printSchema()
pedestrian_count_df.printSchema()

#2.2 Analysis
#Question 1
pedestrian_count_df \
    .where(isnull(pedestrian_count_df.Year) == False) \
    .groupBy("Year").agg(sum("Hourly_Counts").alias("total_pedestrian"), avg("Hourly_Counts").alias("average_pedestrian")) \
    .orderBy( "Year") \
    .show()

#Question 2
pedestrian_count_df.groupBy("Sensor_ID") \
    .agg(avg("Hourly_Counts").alias("average_pedestrian")) \
    .orderBy("average_pedestrian", ascending=False) \
    .limit(5) \
    .join(sensor_location_df, pedestrian_count_df.Sensor_ID == sensor_location_df.sensor_id) \
    .drop(pedestrian_count_df.Sensor_ID) \
    .select("Sensor_ID", "sensor_description", "average_pedestrian") \
    .orderBy("average_pedestrian", ascending=False).show()

#Question 3
breakdown_of_daily_pedestrian_count = pedestrian_count_df \
    .where(pedestrian_count_df.Date_Time.between("2019-07-01", "2019-07-28")) \
    .select("Hourly_Counts", weekofyear(pedestrian_count_df.Date_Time).alias('Week'), dayofweek(pedestrian_count_df.Date_Time).alias('DayOfWeek')) \
    .groupBy("Week", "DayOfWeek") \
    .sum("Hourly_Counts")

breakdown_of_daily_pedestrian_count = breakdown_of_daily_pedestrian_count \
    .withColumn("DayOfWeek", when(breakdown_of_daily_pedestrian_count.DayOfWeek == 1, 7).otherwise(breakdown_of_daily_pedestrian_count.DayOfWeek - 1))

breakdown_of_daily_pedestrian_count \
    .groupBy("Week") \
    .sum("sum(Hourly_Counts)").alias("count") \
    .select(col("Week"), lit("Subtotal").alias("DayOfWeek"), col("sum(sum(Hourly_Counts))").alias("count")) \
    .union(breakdown_of_daily_pedestrian_count) \
    .orderBy("Week", "count", "DayOfWeek") \
    .show()

#Question 4
my_window = Window.partitionBy("Year", "Month", "Mdate").orderBy("Time")
df = pedestrian_count_df.where(pedestrian_count_df.Sensor_ID == 4).where(pedestrian_count_df.Time.between(12, 23)) \
    .withColumn("Prev_Hourly_Counts", lag(pedestrian_count_df.Hourly_Counts).over(my_window))
df = df \
    .withColumn("state", when(isnull(df.Hourly_Counts - df.Prev_Hourly_Counts), "increase") \
    .when(df.Hourly_Counts - df.Prev_Hourly_Counts >= 0, "increase") \
    .otherwise("decrease"))

df.createOrReplaceTempView("pedestrian_count")
df = spark.sql("SELECT Year, Month, Mdate, COLLECT_LIST(state) AS state FROM pedestrian_count GROUP BY Year, Month, Mdate")
df = df.withColumn("state", concat_ws(",", "state"))
df.where(df.state == "increase,increase,increase,decrease,decrease,decrease,decrease,increase,increase,increase,decrease,decrease") \
    .select("Year", "Month", "Mdate") \
    .show()