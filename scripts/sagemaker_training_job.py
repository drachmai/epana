import argparse
import os
import tarfile

import boto3
from sagemaker.pytorch import PyTorch

def extract_tar_gz(tar_gz_path, output_path):
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        tar.extractall(path=output_path)


def args_to_dict(args, excluded):
    return {k: v for k, v in vars(args).items() if k not in excluded and v is not None}


def launch(s3_dataset_uri, hyperparams, download_model_path, instance_type):

    boto3.setup_default_session(region_name='us-west-2')

    # Configure the SageMaker training job
    estimator = PyTorch(
        entry_point="trainers/sagemaker.py",
        source_dir="epana_modeling",
        output_path=f"s3://{os.environ.get('MODEL_BUCKET')}",
        role=os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN"),
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.0",
        py_version="py310",
        hyperparameters=hyperparams,
        environment={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY"),
            "PYTHONPATH": "/opt/ml/code/epana_modeling"
        },
    )
    estimator.fit({"dataset": s3_dataset_uri})

    if download_model_path:
        os.makedirs(download_model_path, exist_ok=True)
        job_name = estimator.latest_training_job.name
        s3 = boto3.client('s3')
        local_tar_gz_path = os.path.join(download_model_path, f"{job_name}.tar.gz")

        s3.download_file(os.environ.get("MODEL_BUCKET"), f"{job_name}/output/model.tar.gz", local_tar_gz_path)
        extract_tar_gz(local_tar_gz_path, download_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_dataset_uri', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--mode', type=str, default='classification')
    parser.add_argument('--focal_alpha_positive', type=float)
    parser.add_argument('--focal_alpha_negative', type=float)
    parser.add_argument('--focal_gamma', type=float)
    parser.add_argument('--accumulation_steps', type=int)
    parser.add_argument('--strategy', type=str)
    parser.add_argument('--embedder_name', type=str)
    parser.add_argument('--instance_type', type=str, defaul='ml.p3.2xlarge')
    parser.add_argument('--download_model_path', type=str, default=None)
    args = parser.parse_args()

    s3_dataset_uri = args.s3_dataset_uri
    download_model_path = args.download_model_path
    instnace_type = args.instance_type
    hyperparams = args_to_dict(args, ['s3_dataset_uri', 'download_model_path', 'instance_type'])

    launch(s3_dataset_uri, hyperparams, download_model_path, instnace_type)