import os
import logging
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, SparkPythonTask, GitSource, TaskDependency, JobCluster

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_databricks_dag():
    """Generates a Databricks Job Workflow DAG (Infrastructure as Code)"""
    try:
        w = WorkspaceClient()
        logger.info("Successfully connected to Databricks Workspace.")
    except Exception as e:
        logger.error(f"Failed to connect. Check DATABRICKS_HOST in .env: {e}")
        return
        
    job_name = "KIE_GenOps_Daily_Pipeline"
    logger.info(f"Building DAG Job: {job_name} ...")

    # Define a shared cluster for all tasks to optimize cost/time
    shared_cluster_key = "kie_pipeline_cluster"
    cluster_spec = JobCluster(
        job_cluster_key=shared_cluster_key,
        new_cluster={
            "spark_version": "14.3.x-cpu-ml-scala2.12", # ML Runtime
            "node_type_id": "Standard_DS3_v2",          # Azure Default Small Node
            "num_workers": 1,
            "spark_env_vars": {
                # We inject our Unity Catalog settings into the Cluster Environment!
                "UC_CATALOG": os.getenv("UC_CATALOG", "main"),
                "UC_SCHEMA": os.getenv("UC_SCHEMA", "default"),
                "UC_MODEL_NAME": os.getenv("UC_MODEL_NAME", "kie_pipeline_model"),
                "MODEL_URI": os.getenv("MODEL_URI", "")
            }
        }
    )
    
    # ------------------ DEFINE DAG TASKS ------------------
    
    # Task 1: Ingestion
    task_ingest = Task(
        task_key="Provision_Data",
        description="Pulls images and metadata into Unity Catalog Volumes.",
        job_cluster_key=shared_cluster_key,
        spark_python_task=SparkPythonTask(
            python_file="upload_dataset.py", 
            source="GIT"
        )
    )

    # Task 2: Model Deployment (Wait for Ingestion)
    task_deploy = Task(
        task_key="Deploy_PyFunc_Model",
        description="Registers the core AI extraction logic to Unity Catalog.",
        depends_on=[TaskDependency(task_key="Provision_Data")],
        job_cluster_key=shared_cluster_key,
        spark_python_task=SparkPythonTask(
            python_file="deploy_model.py", 
            source="GIT"
        )
    )
    
    # Task 3: Batch Inference (Wait for Model Deployment)
    task_infer = Task(
        task_key="Run_Batch_Inference",
        description="Executes the main pipeline against unseen volumetric data.",
        depends_on=[TaskDependency(task_key="Deploy_PyFunc_Model")],
        job_cluster_key=shared_cluster_key,
        spark_python_task=SparkPythonTask(
            python_file="main.py", 
            parameters=["--mode", "inference"],
            source="GIT"
        )
    )

    # ------------------ CREATE THE JOB ------------------
    try:
        # We point Databricks natively to your newly published GitHub repo!
        git_source = GitSource(
            git_url="https://github.com/watthsup/mlflow-genpos-poc.git",
            git_provider="gitHub",
            git_branch="main"
        )

        job = w.jobs.create(
            name=job_name,
            git_source=git_source,
            job_clusters=[cluster_spec],
            tasks=[task_ingest, task_deploy, task_infer]
        )
        logger.info(f"SUCCESS! Created Databricks Workflow Job ID: {job.job_id}")
        logger.info("Go to Databricks UI -> Workflows to see and run your new DAG!")
        
    except Exception as e:
        logger.error(f"Failed to create Job DAG: {e}")

if __name__ == "__main__":
    create_databricks_dag()
