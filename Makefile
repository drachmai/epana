DEFAULT_S3_BUCKET := my-default-bucket
DEFAULT_S3_PATH := my-default-path

LABELED_DIRECTORIES = labeled_datasets/marketing labeled_datasets/medicine labeled_datasets/safety\ services

.PHONY: run-script

label:
ifndef LABELER
	@echo "Please provide a script to run with: make run-script SCRIPT=your_script.py"
else
	@echo "Running $(LABELER) labeler"
	pipenv run python label_$(LABELER).py
endif

create-dataset:
	$(eval S3_BUCKET := $(or $(S3_BUCKET),$(DEFAULT_S3_BUCKET)))
	$(eval S3_PATH := $(or $(S3_PATH),$(DEFAULT_S3_PATH)))
	@echo "Creating dataset in bucket: $(S3_BUCKET), path: $(S3_PATH)"
	pipenv run python dataset_creation/create_dataset.py $(LABELED_DIRECTORIES) --s3-bucket-dest $(S3_BUCKET) --s3-dataset-name $(DEFAULT_S3_BUCKET)
