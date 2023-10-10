from typing import List

from model import ProductSearchWrapper

import chromadb
import langchain
from chromadb.api.segment import API
from langchain.document_loaders import DataFrameLoader
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas
import sentence_transformers
from sentence_transformers import util
import torch


def get_cos_similarity(a, b, mean=False) -> float:
    r = util.pairwise_cos_sim(a, b)
    if mean:
        return torch.mean(r).item()
    return r


def get_corr_coeff(a, b) -> float:
    return np.corrcoef(a, b)[0, 1]


def register_model_mlflow(
    mlflow_client: MlflowClient, model_name: str, artifacts: dict
) -> None:
    # get base environment configuration
    conda_env = mlflow.pyfunc.get_default_conda_env()

    # define packages required by model
    packages = [
        f"pandas=={pandas.__version__}",
        f"langchain=={langchain.__version__}",
        f"chromadb=={chromadb.__version__}",
        f"sentence_transformers=={sentence_transformers.__version__}",
    ]

    # add required packages to environment configuration
    conda_env["dependencies"][-1]["pip"] += packages

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ProductSearchWrapper(),
        conda_env=conda_env,
        artifacts=artifacts,
        registered_model_name=model_name,
    )

    # elevate model to production
    latest_version = mlflow_client.get_latest_versions(model_name, stages=["None"])[
        0
    ].version

    mlflow_client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True,
    )


def batch_chromadb_insertions(
    chroma_client: API, documents: List[Document]
) -> List[Document]:
    """Function borrowed from https://github.com/imartinez/privateGPT/blob/78d1ef44adea1b72235a4cb603bbf0e4d9033d10/ingest.py#L134"""
    max_batch_size = chroma_client.max_batch_size
    for i in range(0, len(documents), max_batch_size):
        yield documents[i : i + max_batch_size]


def does_vectorstore_exist(
    persist_directory: str, embeddings: HuggingFaceEmbeddings
) -> bool:
    """Function borrowed from https://github.com/imartinez/privateGPT/blob/78d1ef44adea1b72235a4cb603bbf0e4d9033d10/ingest.py#L144"""
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    if not db.get()["documents"]:
        return False
    return True


def create_store_embeddings(
    documents: DataFrameLoader,
    model: HuggingFaceEmbeddings,
    chroma: API,
    chroma_path: str,
) -> None:
    batched = batch_chromadb_insertions(chroma, documents)

    # embeddings storage
    if does_vectorstore_exist(chroma_path, model):
        """Borrowed from https://github.com/imartinez/privateGPT/blob/78d1ef44adea1b72235a4cb603bbf0e4d9033d10/ingest.py#L159"""
        db = Chroma(
            persist_directory=chroma_path,
            embedding_function=model,
            client=chroma,
        )
        for batched_chromadb_insertion in batch_chromadb_insertions(chroma, documents):
            db.add_documents(batched_chromadb_insertion)
    else:
        # Create and store locally vectorstore
        # Create the db with the first batch of documents to insert
        batched = batch_chromadb_insertions(chroma, documents)
        first_insertion = next(batched)
        db = Chroma.from_documents(
            first_insertion,
            model,
            persist_directory=chroma_path,
            client=chroma,
        )
        # Add the rest of batches of documents
        for batched_chromadb_insertion in batched:
            db.add_documents(batched_chromadb_insertion)
