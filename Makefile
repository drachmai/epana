include .env
export $(shell sed 's/=.*//' .env)

LABELED_DIRECTORIES = labeled_datasets/marketing labeled_datasets/medicine labeled_datasets/safety\ services
DEFAULT_TRAINING_INSTANCE = ''

setup:
	pipenv install
	pipenv run create_dotenv

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
	AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) TRAINING_DATA_BUCKET_NAME=$(TRAINING_DATA_BUCKET) MODEL_DATA_BUCKET_NAME=$(MODEL_BUCKET) pulumi up -y

destroy-infrastructure:
	cd infrastructure/aws && \
	AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) TRAINING_DATA_BUCKET_NAME=$(TRAINING_DATA_BUCKET) pulumi down -y

train-local:
	PYTHONPATH=epana_modeling pipenv run python epana_modeling/trainers/local.py --dataset_path dataset_dicts/varied-task-concern --model_save_path .

train-sagemaker:
ifndef DOWNLOAD_MODEL_PATH
	pipenv run pip freeze > epana_modeling/requirements.txt
	AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) AWS_REGION=$(AWS_REGION) WANDB_API_KEY=$(WANDB_API_KEY) SAGEMAKER_EXECUTION_ROLE_ARN=$(SAGEMAKER_EXECUTION_ROLE_ARN) PYTHONPATH=scripts pipenv run train_sagemaker --s3_dataset_uri $(DATASET_S3_URI)
else
	pipenv run pip freeze > epana_modeling/requirements.txt
	AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) AWS_REGION=$(AWS_REGION) WANDB_API_KEY=$(WANDB_API_KEY) SAGEMAKER_EXECUTION_ROLE_ARN=$(SAGEMAKER_EXECUTION_ROLE_ARN) PYTHONPATH=scripts pipenv run train_sagemaker --s3_dataset_uri $(DATASET_S3_URI) --download_model_path $(DOWNLOAD_MODEL_PATH)
endif

upload-model-huggingface:
	pipenv run huggingface-cli login --token $(HUGGINGFACE_WRITE_TOKEN)
	pipenv run upload_model_huggingface --model_source_dir $(MODEL_SOURCE_DIR) --huggingface_org $(HUGGINGFACE_ORG) --repo_name $(REPO_NAME)
