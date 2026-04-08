import os
import sys
import logging
import argparse

# Add parent dir to path so it can be ran directly or from airflow/databricks jobs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

from src.doc_guru.inference import run_batch_inference_pipeline
from src.doc_guru.data_loader import fetch_volume_dataset
from src.doc_guru.utils import setup_mlflow, load_config

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Databricks Inference DAG Entrypoint")
    parser.add_argument("--file_path", type=str, default=None, help="Optional specific Databricks Volume path to process from Azure Storage trigger")
    args, unknown = parser.parse_known_args()
    
    logger.info("==== Starting Databricks DAG: Batch Inference ====")
    setup_mlflow()
    model_uri = load_config()
    
    if args.file_path:
        logger.info(f"Triggered by Azure Storage Event! Processing file: {args.file_path}")
        unseen_images = [args.file_path]
    else:
        logger.info("No explicit file_path provided. Fetching batch paths from Volumes...")
        unseen_images = fetch_volume_dataset(mode="inference")
        
    logger.info("Triggering Model Pipeline for Extraction...")
    run_batch_inference_pipeline(
        model_uri=model_uri,
        unseen_images=unseen_images
    )
    logger.info("==== Inference DAG Complete ====")
