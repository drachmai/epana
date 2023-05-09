import json
import argparse
import os
import datetime

now = datetime.datetime.now()

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value
from datasets.filesystems import S3FileSystem

def create_dataset(input_paths, dataset_name, local_dir, train_sample_size=None, val_sample_size=None, test_sample_size=None, s3_bucket=None, s3_dir=None, huggingface_org_name=None):
    columns_to_keep = ['previous_chat', 'last_message', 'concerning_definitions', 'is_concerning_score']

    train_df, val_df, test_df = process_datasets(input_paths, train_sample_size, val_sample_size, test_sample_size)

    train_df = train_df[columns_to_keep].reset_index(drop=True)
    val_df = val_df[columns_to_keep].reset_index(drop=True)
    test_df = test_df[columns_to_keep].reset_index(drop=True)

    features = Features({
        'previous_chat': Value('string'),
        'last_message': Value('string'),
        'concerning_definitions': Value('string'),
        'is_concerning_score': Value('float32')
    })

    train_dataset = Dataset.from_pandas(train_df, features=features)
    val_dataset = Dataset.from_pandas(val_df, features=features)
    test_dataset = Dataset.from_pandas(test_df, features=features)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    local_path = f"{local_dir}/{dataset_name}"

    dataset_dict.save_to_disk(local_path)

    if s3_bucket:

        if not s3_dir:
            s3_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY")
        aws_secret_access_key = os.environ.get("AWS_SECRET_KEY")

        s3 = S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
        dataset_dict.save_to_disk(f"s3://{s3_bucket}/{s3_dir}/{dataset_name}", storage_options=s3.storage_options)

    if huggingface_org_name:
        dataset_dict.push_to_hub(f"{huggingface_org_name}/{dataset_name}", token="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")

    return dataset_dict

def process_dataset(data, metadata):
    data['concerning_label'] = data['label'].str.split("\n").apply(lambda x: get_label_value(x, "Label -"))
    data['explanation_label'] = data['label'].str.split("\n").apply(lambda x: get_label_value(x, "Explanation -"))
    data['confidence_label'] = data['label'].str.split("\n").apply(lambda x: get_label_value(x, "Confidence -"))

    data = data.dropna()

    #Sometimes the labels indicate they can't make a decision, drop these
    data = data[data['concerning_label'].isin(["concerning", "not concerning"])]
    data['confidence_label'] = data['confidence_label'].str.extract('(\d*\.?\d+)')
    data = data.dropna()

    data['concerning_definitions'] = metadata['concerning_definitions']
    data['is_concerning_score'] = data['confidence_label'].astype(float) * data['concerning_label'].apply(convert_label_to_number)

    train_data = data.sample(frac=0.8,random_state=200)
    test_val_data=data.drop(train_data.index)
    test_data = test_val_data.sample(frac=0.5, random_state=200)
    val_data = test_val_data.drop(test_data.index)

    return train_data, val_data, test_data


def process_datasets(paths, train_sample_size=None, val_sample_size=None, test_sample_size=None):
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for path in paths:
        dataset_path = os.path.join(path, "dataset.jsonl")
        metadata_path = os.path.join(path, "metadata.json")

        data = pd.read_json(dataset_path, lines=True)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        train, val, test = process_dataset(data, metadata)
        train_datasets.append(train)
        val_datasets.append(val)
        test_datasets.append(test)

    train_dataset = pd.concat(train_datasets)
    val_dataset = pd.concat(val_datasets)
    test_dataset = pd.concat(test_datasets)

    if train_sample_size:
        train_dataset = train_dataset.sample(int(train_sample_size))
    if val_sample_size:
        val_dataset = val_dataset.sample(int(val_sample_size))
    if test_sample_size:
        test_dataset = test_dataset.sample(int(test_sample_size))

    return train_dataset, val_dataset, test_dataset


def get_label_value(strings, prefix):
    elements = [e.strip().lstrip(prefix) for e in strings if e.strip().startswith(prefix)]
    if elements:
        return elements[0]
    else:
        return None
     
       
def convert_label_to_number(label):
    if label.lower().strip() == "concerning":
        return 1
    elif label.lower().strip() == "not concerning":
        return -1
    else:
        raise ValueError(f"Invalid label {label}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('input_files', metavar='input_file', type=str, nargs='+',
                        help='List of input file paths')
    parser.add_argument('--dataset-name', dest='dataset_name', type=str,
                    help='Name for the dataset')
    parser.add_argument('--local-dir', dest='local_dir', type=str,
                    help='Destination of dataset locally')
    parser.add_argument('--train-sample-size', dest='train_sample_size', type=float,
                    help='Size of the training set', default=None)
    parser.add_argument('--val-sample-size', dest='val_sample_size', type=float,
                        help='Size of the validation set', default=None)
    parser.add_argument('--test-sample-size', dest='test_sample_size', type=float,
                        help='Size of the test set', default=None)
    parser.add_argument('--s3-dir', dest='s3_dir', type=str,
                    help='Optional directory to put dataset in', default=None)
    parser.add_argument('--s3-bucket', dest='s3_bucket', type=str,
                    help='Bucket to putdataset in', default=None)
    parser.add_argument('--huggingface-org-name', dest='huggingface_org_name', type=str,
                    help='Huggingface org to push to', default=None)
    
    args = parser.parse_args()
    input_paths = args.input_files
    train_sample_size = args.train_sample_size
    val_sample_size = args.val_sample_size
    test_sample_size = args.test_sample_size
    s3_bucket = args.s3_bucket
    s3_dir = args.s3_dir
    dataset_name = args.dataset_name
    local_dir = args.local_dir
    huggingface_org_name = args.huggingface_org_name

    create_dataset(input_paths, dataset_name, local_dir, train_sample_size, val_sample_size, test_sample_size, s3_bucket, s3_dir, huggingface_org_name)