import os
import json

import pulumi
from pulumi_aws import iam
from pulumi_aws import s3

# Function to update the env file
def update_env_file(key, value):
    env_file = os.path.join("..", "..", ".env")
    env_vars = {}

    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if "=" in line:
                    name, val = line.strip().split("=", 1)
                    env_vars[name] = val

    env_vars[key] = value

    with open(env_file, "w") as f:
        for name, val in env_vars.items():
            f.write(f"{name}={val}\n")

# Create an S3 bucket for training data
training_data_bucket = s3.Bucket('training-data-bucket', bucket=os.environ.get("TRAINING_DATA_BUCKET_NAME"))

# Create an S3 bucket for SageMaker models
model_bucket = s3.Bucket('model-data-bucket', bucket=os.environ.get("MODEL_DATA_BUCKET_NAME"))

# Execution role to run sagemaker
sagemaker_execution_role = iam.Role(
    'SageMakerExecutionRole',
    assume_role_policy=pulumi.Output.from_input({
        'Version': '2012-10-17',
        'Statement': [
            {
                'Effect': 'Allow',
                'Principal': {
                    'Service': 'sagemaker.amazonaws.com'
                },
                'Action': 'sts:AssumeRole'
            }
        ]
    }).apply(lambda v: json.dumps(v)),
)

sagemaker_full_access_policy = iam.RolePolicyAttachment(
    'SageMakerFullAccess',
    policy_arn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
    role=sagemaker_execution_role.name,
)

# Policy to allow sagemaker to read and write S3
s3_bucket_access_policy = iam.Policy("S3BucketAccessPolicy",
    policy=pulumi.Output.all(training_data_bucket.arn, training_data_bucket.arn.apply(lambda arn: f"{arn}/*"), model_bucket.arn, model_bucket.arn.apply(lambda arn: f"{arn}/*")).apply(lambda args: json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:ListBucket",
                    "s3:PutObject"
                ],
                "Resource": [
                    args[0],  # Training bucket ARN
                    args[1],  # Training bucket objects ARN
                    args[2],  # Model bucket ARN
                    args[3],  # Model bucket objects ARN
                ]
            }
        ]
    })),
)

iam.RolePolicyAttachment("S3BucketAccessPolicyAttachment",
    policy_arn=s3_bucket_access_policy.arn,
    role=sagemaker_execution_role.name
)

# Update the environment
sagemaker_execution_role.arn.apply(lambda arn: update_env_file("SAGEMAKER_EXECUTION_ROLE_ARN", arn))
training_data_bucket.id.apply(lambda bucket_id: pulumi.export('training_data_bucket', bucket_id))
model_bucket.id.apply(lambda bucket_id: pulumi.export('model_data_bucket', bucket_id))
