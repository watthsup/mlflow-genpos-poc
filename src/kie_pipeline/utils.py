import os
import sys
import logging
import mlflow

logger = logging.getLogger(__name__)

def load_config() -> str:
    """Reads the dynamically created model_uri from the .env file."""
    model_uri = os.getenv("MODEL_URI")
    if not model_uri:
        logger.error("MODEL_URI not found in environment! Please run `deploy_model.py` first to register the model.")
        sys.exit(1)
    return model_uri

def setup_mlflow():
    """Initializes standard MLflow connection configuration from environment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri("databricks-uc")
    try:
        if hasattr(mlflow, "langchain"):
            mlflow.langchain.autolog()
    except Exception:
        pass
