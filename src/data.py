import os

from config import config

from pandas import DataFrame
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, when

# All paths where to save Parquet files
product_parquet_path = os.path.join(config["database_path"], "product.parquet")
query_parquet_path = os.path.join(config["database_path"], "query.parquet")
label_parquet_path = os.path.join(config["database_path"], "label.parquet")

# Tables schemas
products_schema = StructType(
    [
        StructField("product_id", IntegerType()),
        StructField("product_name", StringType()),
        StructField("product_class", StringType()),
        StructField("category hierarchy", StringType()),
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


def load_data_spark(spark: SparkSession):
    # process products
    (
        spark.read.csv(
            path=os.path.join(config["dataset_path"], "product.csv"),
            sep="\t",
            header=True,
            schema=products_schema,
        )
        .write.mode("overwrite")
        .parquet(product_parquet_path)
    )

    # process queries
    (
        spark.read.csv(
            path=os.path.join(config["dataset_path"], "query.csv"),
            sep="\t",
            header=True,
            schema=queries_schema,
        )
        .write.mode("overwrite")
        .parquet(query_parquet_path)
    )

    # process labels and add "label_score"
    labels_df = spark.read.csv(
        path=os.path.join(config["dataset_path"], "label.csv"),
        sep="\t",
        header=True,
        schema=labels_schema,
    )
    labels_df = labels_df.withColumn(
        "label_score",
        when(lower(col("label")) == "exact", 1.0).otherwise(
            when(lower(col("label")) == "partial", 0.75).otherwise(
                when(lower(col("label")) == "irrelevant", 0.0).otherwise(0.0)
            )
        ),
    )
    labels_df.write.mode("overwrite").parquet(label_parquet_path)


def get_search_df(spark: SparkSession) -> DataFrame:
    products_df = spark.read.parquet(product_parquet_path)
    labels_df = spark.read.parquet(label_parquet_path)
    queries_df = spark.read.parquet(query_parquet_path)

    return (
        products_df.selectExpr(
            "product_id",
            "product_name",
            "COALESCE(product_description, product_name) as product_text",
        )
        .join(labels_df, on="product_id")
        .join(queries_df, on="query_id")
        .selectExpr("query", "product_text", "label_score as score")
    ).toPandas()
