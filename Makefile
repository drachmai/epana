include .env
export $(shell sed 's/=.*//' envfile)

LABELED_DIRECTORIES = labeled_datasets/marketing labeled_datasets/medicine labeled_datasets/safety\ services

install:
	pipenv install

label:
ifndef LABELER
	@echo "Please provide a labeler to use: make label LABELER=marketing"
else
	@echo "Running $(LABELER) labeler"
	pipenv run python label_$(LABELER).py
endif

create-dataset:
	pipenv run python dataset_creation/create_dataset.py $(LABELED_DIRECTORIES) --dataset-name $(DATASET_NAME) --local-dir $(LOCAL_DATASET_PATH) --s3-bucket $(TRAINING_DATA_BUCKET) --huggingface-org-name $(HUGGINGFACE_ORG_NAME)

create-infrastructure:
	cd infrastructure/aws && \
	AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY) AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_KEY) TRAINING_DATA_BUCKET_NAME=$(TRAINING_DATA_BUCKET) pulumi up -y

destroy-infrastructure:
	cd infrastructure/aws && \
	AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY) AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_KEY) TRAINING_DATA_BUCKET_NAME=$(TRAINING_DATA_BUCKET) pulumi down -y

train-local:
	PYTHONPATH=modeling pipenv run python modeling/trainers/local.py --dataset_path dataset_dicts/varied-task-concern --model_save_path .

train-sagemaker:
ifndef DATASET_S3_URI
	@echo "Please specify an S3 URI for the DatasetDict to use for training: make train-sagemaker DATASET_S3_URI=s3://dataset/path"
	pipenv lock -r > modeling/requirements.txt
	AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY) AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_KEY) DATASET_S3_URI=$(DATASET_S3_URI) WANDB_API_KEY=$(WANDB_API_KEY) pipenv run scripts/sagemaker_training_job.py
