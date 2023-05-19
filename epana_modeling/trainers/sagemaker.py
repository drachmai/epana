import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import argparse

from datasets import load_from_disk

from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--mode', type=str, default='classification')
    parser.add_argument('--focal_alpha', type=float, default=0.2)
    parser.add_argument('--focal_gamma', type=float, default=3.0)
    parser.add_argument('--accumulation_steps', type=int, default=16)
    parser.add_argument('--strategy', type=str, default='higher_level_activations')
    parser.add_argument('--embedder_name', type=str, default='distilroberta-base')
    args = parser.parse_args()

    # Where the final model will be saved
    model_save_path = os.environ['SM_MODEL_DIR']

    # Where the training dataset will be
    dataset_path = os.environ['SM_CHANNEL_DATASET']
    dataset = load_from_disk(dataset_path)

    # Update the training script to use the command-line arguments
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    step_size = args.step_size
    gamma = args.gamma
    mode = args.mode
    focal_alpha = args.focal_alpha
    focal_gamma = args.focal_gamma
    accumulation_steps = args.accumulation_steps
    strategy = args.strategy
    embedder_name = args.embedder_name

    model, tokenizer = train(num_epochs, batch_size, learning_rate, step_size, gamma, embedder_name, dataset, accumulation_steps, strategy, mode, focal_alpha, focal_gamma, train_sample_size=None, val_sample_size=None, test_sample_size=None)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
