import os
import sys
import logging
import argparse

# Add parent dir to path so it can be ran directly or from airflow/databricks jobs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

from src.doc_guru.evaluation import run_evaluation_pipeline
from src.doc_guru.data_loader import fetch_volume_dataset
from src.doc_guru.utils import setup_mlflow, load_config

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Databricks Evaluation DAG Entrypoint")
    args, unknown = parser.parse_known_args()
    
    logger.info("==== Starting Databricks DAG: Evaluation Pipeline ====")
    setup_mlflow()
    model_uri = load_config()
    
    logger.info("Fetching Ground Truth JSON from Databricks Volumes...")
    gt_dataset = fetch_volume_dataset(mode="evaluate")
    
    if not gt_dataset:
        logger.error("Empty evaluation dataset. Ensure upload_dataset.py seeded the volumes.")
        exit(1)
        
    logger.info("Triggering Evaluation Core Application...")
    run_evaluation_pipeline(
        model_uri=model_uri, 
        dataset=gt_dataset, 
        prompt_ver="v1.2-alpha-eval"
    )
    
    logger.info("==== Evaluation DAG Complete ====")
