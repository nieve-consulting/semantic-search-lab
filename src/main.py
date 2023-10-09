from config import config
from data_load import load_data
from embeddings import create_embeddings
from register_mlflow import register_mlflow

from chromadb import chromadb
import mlflow
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("app").getOrCreate()

    # set current datebase context
    _ = spark.sql("create database if not exists {0}".format(config["database"]))
    _ = spark.catalog.setCurrentDatabase(config["database"])

    # load data
    load_data(spark)

    # chroma DB client
    chroma = chromadb.PersistentClient(path=config["chromadb_path"])

    # create embeddings
    create_embeddings(spark, chroma)

    artifacts = {
        "embedding_model": config["embedding_model_path"],
        "chromadb": config["chromadb_path"],
    }

    # register model and artifacts to mlflow
    mlflow.set_experiment(config["mlflow_experiment"])
    mlflow_client = mlflow.MlflowClient()

    register_mlflow(mlflow_client, artifacts)
