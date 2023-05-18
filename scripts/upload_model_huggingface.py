import argparse

from huggingface_hub import HfApi

def upload_model(model_source_dir, huggingface_org, repo_name):
    api = HfApi()
    api.upload_folder(
        folder_path=model_source_dir,
        repo_id=f"{huggingface_org}/{repo_name}",
        repo_type="model",
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_source_dir', type=str)
    parser.add_argument('--huggingface_org', type=str)
    parser.add_argument('--repo_name', type=str)

    args = parser.parse_args()

    model_source_dir = args.model_source_dir
    huggingface_org = args.huggingface_org
    repo_name = args.repo_name

    upload_model(model_source_dir, huggingface_org, repo_name)
