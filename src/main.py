from config import config
from data import load_data_spark, get_search_df
from utils import (
    create_store_embeddings,
    get_cos_similarity,
    get_corr_coeff,
    register_model_mlflow,
)

from chromadb import chromadb
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
import mlflow
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer
import torch

if __name__ == "__main__":
    spark = SparkSession.builder.appName("app").getOrCreate()

    # load data to spark db
    load_data_spark(spark)

    chroma = chromadb.PersistentClient(path=config["chromadb_path"])

    # assemble product text relevant to search
    product_text_pd = get_search_df(spark)

    # download embeddings model
    model = SentenceTransformer("all-MiniLM-L12-v2")

    # reload model using langchain wrapper
    model.save(config["embedding_model_path"])
    embedding_model = HuggingFaceEmbeddings(model_name=config["embedding_model_path"])

    # assemble product documents in required format (id, text)
    documents = DataFrameLoader(
        product_text_pd, page_content_column="product_text"
    ).load()

    # store embeddings in chroma
    create_store_embeddings(documents, embedding_model, chroma, config["chromadb_path"])

    # register model and artifacts to mlflow
    mlflow_client = mlflow.MlflowClient()
    mlflow.set_experiment(config["mlflow_experiment"])

    # add eval. metrics to mlflow run
    query_embeddings = model.encode(product_text_pd["query"].tolist())
    product_embeddings = model.encode(product_text_pd["product_text"].tolist())

    cos_sim = get_cos_similarity(query_embeddings, product_embeddings)
    mean_cos_sim = torch.mean(cos_sim).item()
    corr_coef = get_corr_coeff(cos_sim, product_text_pd["score"].values)

    with mlflow.start_run() as run:
        mlflow.log_metric("cos_sim", mean_cos_sim)
        mlflow.log_metric("corr_coef", corr_coef)

        register_model_mlflow(
            mlflow_client,
            config["basic_model_name"],
            {
                "embedding_model": config["embedding_model_path"],
                "chromadb": config["chromadb_path"],
            },
        )

        mlflow.end_run()
