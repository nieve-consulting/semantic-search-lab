import os

from dotenv import load_dotenv

load_dotenv()

config = {}

config["database"] = os.getenv("SPARK_DATABASE")
config["database_path"] = os.getenv("SPARK_DATABASE_PATH")
config["mlflow_experiment"] = os.getenv("MLFLOW_EXPERIMENT")
config["mlflow_registry_path"] = os.getenv("MLFLOW_REGISTRY_PATH")
config["basic_model_name"] = os.getenv("BASIC_MODEL_NAME")
config["tuned_model_name"] = os.getenv("TUNED_MODEL_NAME")
config["dataset_path"] = os.getenv("DATASET_PATH")
config["embedding_model_path"] = os.getenv("EMBEDDING_MODEL_PATH")
config["tuned_embedding_model_path"] = os.getenv("TUNED_EMBEDDING_MODEL_PATH")
config["chromadb_path"] = os.getenv("CHROMADB_PATH")
config["tuned_chromadb_path"] = os.getenv("TUNED_CHROMADB_PATH")
