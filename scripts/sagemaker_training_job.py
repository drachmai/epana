import argparse
import os

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

def launch(s3_dataset_uri):

    boto3.setup_default_session(region_name='us-west-2')

    # Configure the SageMaker training job
    estimator = PyTorch(
        entry_point="trainers/sagemaker.py",
        source_dir="modeling",
        role=os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN"),
        instance_type="ml.g4dn.2xlarge",
        instance_count=1,
        framework_version="2.0",
        py_version="py310",
        hyperparameters={},
        environment={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY"),
            "PYTHONPATH": "/opt/ml/code/modeling"
        },
    )
    estimator.fit({"dataset": s3_dataset_uri})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_dataset_uri', type=str)
    args = parser.parse_args()

    # Update the training script to use the command-line arguments
    s3_dataset_uri = args.s3_dataset_uri

    launch(s3_dataset_uri)