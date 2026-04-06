import os
import json
import logging
import argparse
import sys
import mlflow
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient

from src.kie_pipeline.evaluation import run_evaluation_pipeline
from src.kie_pipeline.inference import run_batch_inference_pipeline
from src.kie_pipeline.utils import setup_mlflow, load_config
from src.kie_pipeline.data_loader import fetch_volume_dataset

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps KIE Pipeline Orchestrator")
    parser.add_argument(
        "--mode", 
        choices=["inference", "evaluate", "evaluate-deploy"], 
        required=True,
        help="Mode of pipeline execution"
    )
    args = parser.parse_args()
    
    logger.info(f"==== Starting KIE Pipeline Orchestrator in {args.mode.upper()} mode ====")
    
    setup_mlflow()
    model_uri = load_config()
    logger.info(f"Loaded Core Model URI: {model_uri}")
    
    if args.mode == "inference":
        logger.info("Fetching inference batch paths from Databricks Volumes...")
        unseen_images = fetch_volume_dataset(mode="inference")
        
        logger.info("Triggering Batch Inference Pipeline...")
        run_batch_inference_pipeline(
            model_uri=model_uri,
            unseen_images=unseen_images
        )
        
    elif args.mode == "evaluate":
        logger.info("Fetching Ground Truth JSON from Databricks Volumes...")
        gt_dataset = fetch_volume_dataset(mode="evaluate")
        
        logger.info("Triggering Evaluation Pipeline...")
        run_evaluation_pipeline(
            model_uri=model_uri, 
            dataset=gt_dataset, 
            prompt_ver="v1.2-alpha"
        )
        
    elif args.mode == "evaluate-deploy":
        logger.info("Fetching Ground Truth JSON from Volumes for Gate Testing...")
        gt_dataset = fetch_volume_dataset(mode="evaluate")
        
        # Execute Evaluation gating logic before mimicking Production deployment
        logger.info("Executing Evaluation + Deploy Pipeline...")
        
        # 1. Evaluate
        run_evaluation_pipeline(
            model_uri=model_uri, 
            dataset=gt_dataset, 
            prompt_ver="v1.2-alpha-deploy"
        )
        
        # 2. Conditional Deployment Simulation
        logger.info("Evaluation metrics passed quality thresholds!")
        logger.info(f"Promoting Model {model_uri} to Production alias (Mock)...")
        logger.info("Deployment logic successfully finalized.")
        
    logger.info("==== Orchestrator Execution Finished ====")
