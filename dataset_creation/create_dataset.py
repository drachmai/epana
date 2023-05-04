import json
import argparse
import os

import pandas as pd
import s3fs

def create_dataset(input_paths, s3_bucket, s3_dir_path, train_sample_size=None, val_sample_size=None, test_sample_size=None):
    train_data, val_data, test_data = process_datasets(input_paths, train_sample_size, val_sample_size, test_sample_size)

    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY")
    aws_secret_access_key = os.environ.get("AWS_SECRET_KEY")
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

    with s3.open(f"{s3_bucket}/{s3_dir_path}/train.jsonl", 'w') as f:
        train_data.to_json(f, orient='records', lines=True)
    with s3.open(f"{s3_bucket}/{s3_dir_path}/val.jsonl", 'w') as f:
        val_data.to_json(f, orient='records', lines=True)
    with s3.open(f"{s3_bucket}/{s3_dir_path}/train.jsonl", 'w') as f:
        test_data.to_json(f, orient='records', lines=True)


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
    parser.add_argument('--s3-bucket-dest', dest='s3_bucket_dest', type=str,
                    help='Destination of final datasets')
    parser.add_argument('--s3-dataset-name', dest='s3_dataset_name', type=str,
                    help='Name for the s3 dataset')
    parser.add_argument('--train-sample-size', dest='train_sample_size', type=float,
                    help='Size of the training set', default=None)
    parser.add_argument('--val-sample-size', dest='val_sample_size', type=float,
                        help='Size of the validation set', default=None)
    parser.add_argument('--test-sample-size', dest='test_sample_size', type=float,
                        help='Size of the test set', default=None)
    args = parser.parse_args()
    input_paths = args.input_files
    train_sample_size = args.train_sample_size
    val_sample_size = args.val_sample_size
    test_sample_size = args.test_sample_size
    s3_bucket_dest = args.s3_bucket_dest
    s3_dataset_name = args.s3_dataset_name

    create_dataset(input_paths, s3_bucket_dest, s3_dataset_name, train_sample_size, val_sample_size, test_sample_size)