from config import config
from model import ProductSearchWrapper

import mlflow
import pandas
import langchain
import chromadb
import sentence_transformers


def register_mlflow(mlflow_client, artifacts: dict):
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

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ProductSearchWrapper(),
            conda_env=conda_env,
            artifacts=artifacts,  # items at artifact path will be loaded into mlflow repository
            registered_model_name=config["basic_model_name"],
        )

    # elevate model to production
    latest_version = mlflow_client.get_latest_versions(
        config["basic_model_name"], stages=["None"]
    )[0].version

    mlflow_client.transition_model_version_stage(
        name=config["basic_model_name"],
        version=latest_version,
        stage="Production",
        archive_existing_versions=True,
    )
