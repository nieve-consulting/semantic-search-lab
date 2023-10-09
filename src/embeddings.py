from typing import List

from config import config

from chromadb.api.segment import API
from langchain.document_loaders import DataFrameLoader
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer


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


def create_embeddings(spark: SparkSession, chroma: API):
    # assemble product text relevant to search
    product_text_pd = (
        spark.table("products").selectExpr(
            "product_id",
            "product_name",
            "COALESCE(product_description, product_name) as product_text",
        )
    ).toPandas()

    # download embeddings model
    original_model = SentenceTransformer("all-MiniLM-L12-v2")

    # reload model using langchain wrapper
    original_model.save(config["embedding_model_path"])
    embedding_model = HuggingFaceEmbeddings(model_name=config["embedding_model_path"])

    # assemble product documents in required format (id, text)
    documents = DataFrameLoader(
        product_text_pd, page_content_column="product_text"
    ).load()

    batched = batch_chromadb_insertions(chroma, documents)

    # define logic for embeddings storage
    if does_vectorstore_exist(config["chromadb_path"], embedding_model):
        """Borrowed from https://github.com/imartinez/privateGPT/blob/78d1ef44adea1b72235a4cb603bbf0e4d9033d10/ingest.py#L159"""
        db = Chroma(
            persist_directory=config["chromadb_path"],
            embedding_function=embedding_model,
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
            embedding_model,
            persist_directory=config["chromadb_path"],
            client=chroma,
        )
        # Add the rest of batches of documents
        for batched_chromadb_insertion in batched:
            db.add_documents(batched_chromadb_insertion)
