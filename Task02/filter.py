from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Initialize Spark session
spark = SparkSession.builder.appName("FilterSentiment").getOrCreate()

# Load CSV file
input_path = "Reviews.csv"
df = spark.read.option("header", True).csv(input_path)

# Print columns for debugging
print("Columns:", df.columns)

# Ensure the column exists before proceeding
if "Score" in df.columns:
    # Convert to float, handling non-numeric values
    df = df.withColumn(
        "Score",
        when(col("Score").rlike("^[0-9.]+$"), col("Score").cast("float"))
    )

    # Filter rows where sentiment_score is >= 0.5
    filtered_df = df.filter(col("Score") >= 0.5)

    # Write the filtered data to CSV
    output_path = "filtered_output.csv"
    filtered_df.write.option("header", True).csv(output_path)

    print("Filtered data successfully saved to:", output_path)
else:
    print("Error: 'Score' column not found in CSV!")

# Stop Spark session
spark.stop()
