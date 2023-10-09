import os

from dotenv import load_dotenv

load_dotenv()

config = {}

config["database"] = os.getenv("SPARK_DATABASE")
config["dbfs_path"] = os.getenv("DBFS_PATH")
config["mlflow_experiment"] = os.getenv("MLFLOW_EXPERIMENT")
config["mlflow_registry_path"] = os.getenv("MLFLOW_REGISTRY_PATH")
config["basic_model_name"] = "wands_basic_search"
config["tuned_model_name"] = "wands_tuned_search"
config["dataset_path"] = os.getenv("DATASET_PATH")
config["embedding_model_path"] = os.getenv("EMBEDDING_MODEL_PATH")
config["chromadb_path"] = os.getenv("CHROMADB_PATH")
