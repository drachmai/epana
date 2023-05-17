import os

def create_or_load_env_file():
    env = {}

    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            pass
    else:
        with open('.env', 'r') as f:
            for line in f:
                key, value = line.strip().split('=', 1)
                env[key] = value
    
    return env

def prompt_user_variables(variables, current_env):
    new_env = {}

    for var in variables:
        current_value = current_env.get(var)
        user_input = input(f'{var} [{current_value}]: ')
        if user_input:
            new_env[var] = user_input
        else:
            new_env[var] = current_value
    
    return new_env

def write_env_file(env):
    with open('.env', 'w') as f:
        for var in env:
            value = env.get(var, "")
            if value:
                f.write(f'{var}={value}\n')

if __name__ == '__main__':
    variables = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "SAGEMAKER_EXECUTION_ROLE_ARN",
        "OPENAI_API_KEY",
        "PULUMI_TOKEN",
        "TRAINING_DATA_BUCKET",
        "LOCAL_DATASET_PATH",
        "DATASET_NAME",
        "HUGGINGFACE_ORG_NAME",
        "WANDB_API_KEY",
        "HUGGINGFACE_WRITE_TOKEN"
    ]
    
    current_env = create_or_load_env_file()
    new_env = prompt_user_variables(variables, current_env)
    write_env_file(new_env)