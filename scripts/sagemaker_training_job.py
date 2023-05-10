import argparse
import os

import sagemaker
from sagemaker.pytorch import PyTorch

# Your wandb API key
wandb_api_key = os.environ.get("WANDB_API_KEY")
aws_access_key = os.environ.get("AWS_ACCESS_KEY")
aws_secret_key = os.environ.get("AWS_SECRET_KEY")

def launch():
    # Configure the SageMaker training job
    estimator = PyTorch(
        entry_point="trainers/sagemaker.py",
        source_dir="modeling",
        role=sagemaker.get_execution_role(),
        instance_type="ml.g4dn.2xlarge",
        instance_count=1,
        framework_version="2.0.1",
        py_version="py3",
        hyperparameters={},
        environment={"WANDB_API_KEY": wandb_api_key},
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_dataset_uri', type=str)
    args = parser.parse_args()

    # Update the training script to use the command-line arguments
    s3_dataset_uri = args.s3_dataset_uri

    launch(s3_dataset_uri)