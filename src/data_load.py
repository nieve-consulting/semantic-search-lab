import os

from config import config

import mlflow
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, when

products_schema = StructType(
    [
        StructField("product_id", IntegerType()),
        StructField("product_name", StringType()),
        StructField("product_class", StringType()),
        StructField("category_hierarchy", StringType()),
        StructField("product_description", StringType()),
        StructField("product_features", StringType()),
        StructField("rating_count", FloatType()),
        StructField("average_rating", FloatType()),
        StructField("review_count", FloatType()),
    ]
)

queries_schema = StructType(
    [
        StructField("query_id", IntegerType()),
        StructField("query", StringType()),
        StructField("query_class", StringType()),
    ]
)

labels_schema = StructType(
    [
        StructField("id", IntegerType()),
        StructField("query_id", IntegerType()),
        StructField("product_id", IntegerType()),
        StructField("label", StringType()),
    ]
)


def load_data(spark: SparkSession):
    # process products
    _ = (
        spark.read.csv(
            path=os.path.join(config["dataset_path"], "product.csv"),
            sep="\t",
            header=True,
            schema=products_schema,
        )
        .write.format("parquet")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable("products")
    )

    # process queries
    _ = (
        spark.read.csv(
            path=os.path.join(config["dataset_path"], "query.csv"),
            sep="\t",
            header=True,
            schema=queries_schema,
        )
        .write.format("parquet")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable("queries")
    )

    # process labels
    _ = (
        spark.read.csv(
            path=os.path.join(config["dataset_path"], "label.csv"),
            sep="\t",
            header=True,
            schema=labels_schema,
        )
        .write.format("parquet")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable("labels")
    )

    # assign label scores
    if "label_score" not in spark.table("labels").columns:
        _ = spark.sql("ALTER TABLE labels ADD COLUMN label_score FLOAT")

    # First, load the 'labels' table into a DataFrame
    labels_df = spark.table("labels")

    # Create a new DataFrame with the updated values
    labels_updated_df = labels_df.withColumn(
        "label_score",
        when(lower(col("label")) == "exact", 1.0).otherwise(
            when(lower(col("label")) == "partial", 0.75).otherwise(
                when(lower(col("label")) == "irrelevant", 0.0).otherwise(None)
            )
        ),
    )

    # Save this DataFrame to a temporary table
    labels_updated_df.write.format("parquet").mode("overwrite").saveAsTable(
        "labels_temp"
    )

    # Drop the original table and rename the temporary table
    spark.sql("DROP TABLE IF EXISTS labels")
    spark.sql("ALTER TABLE labels_temp RENAME TO labels")
