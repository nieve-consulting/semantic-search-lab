from config import config
from data import get_search_df, load_data_spark
from utils import (
    get_cos_similarity,
    get_corr_coeff,
    create_store_embeddings,
    register_model_mlflow,
)

from chromadb import chromadb
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import mlflow
import pyspark as spark
from pyspark.sql import SparkSession
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
)
import torch
from torch.utils.data import DataLoader


def create_input(a, b, score):
    return InputExample(texts=[a, b], label=score)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("app").getOrCreate()

    # assemble product text relevant to search
    product_text_pd = get_search_df(spark)

    # download embeddings model
    model = SentenceTransformer("all-MiniLM-L12-v2")

    # define instructions for feeding inputs to model
    inputs = product_text_pd.apply(
        lambda s: create_input(s["query"], s["product_text"], s["score"]), axis=1
    ).to_list()

    input_dataloader = DataLoader(inputs, shuffle=True, batch_size=16)

    # define loss metric to optimize for
    loss = losses.CosineSimilarityLoss(model)

    # tune the model on the input data
    model.fit(train_objectives=[(input_dataloader, loss)], epochs=1, warmup_steps=100)

    # reload model using langchain wrapper
    model.save(config["tuned_embedding_model_path"])
    embedding_model = HuggingFaceEmbeddings(
        model_name=config["tuned_embedding_model_path"]
    )

    chroma = chromadb.PersistentClient(path=config["chromadb_path"])

    # assemble product documents in required format (id, text)
    documents = DataFrameLoader(
        product_text_pd, page_content_column="product_text"
    ).load()

    # store embeddings in chroma
    create_store_embeddings(
        documents, embedding_model, chroma, config["tuned_chromadb_path"]
    )

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
            config["tuned_model_name"],
            {
                "embedding_model": config["tuned_embedding_model_path"],
                "chromadb": config["tuned_chromadb_path"],
            },
        )

        mlflow.end_run()
