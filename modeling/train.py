import argparse
import os
import time

import torch
from transformers import AutoTokenizer, PretrainedConfig
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from datasets import load_dataset

from model import ConcernDataset, CrossAttentionModel


def create_data_loader(data, tokenizer, batch_size):
    dataset = ConcernDataset(data, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    mae_sum = 0

    with torch.no_grad():
        for inputs_1, inputs_2, inputs_3, label in data_loader:
            inputs_1 = {key: value.to(device) for key, value in inputs_1.items()}
            inputs_2 = {key: value.to(device) for key, value in inputs_2.items()}
            inputs_3 = {key: value.to(device) for key, value in inputs_3.items()}
            label = label.to(device)

            logits = model(input_ids_1=inputs_1['input_ids'], attention_mask_1=inputs_1['attention_mask'],
            input_ids_2=inputs_2['input_ids'], attention_mask_2=inputs_2['attention_mask'],
            input_ids_3=inputs_3['input_ids'], attention_mask_3=inputs_3['attention_mask']).unsqueeze(-1)

            loss = criterion(logits, label)
            total_loss += loss.item()

            # Calculate mean absolute error (MAE)
            mae_sum += torch.abs(logits - label).sum().item()

    avg_mae = mae_sum / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    return avg_mae, avg_loss


def train(num_epochs, batch_size, learning_rate, step_size, gamma, embedder_name, dataset_path, model_save_path, accumulation_steps, train_sample_size=None, val_sample_size=None, test_sample_size=None):

    dataset = load_dataset(dataset_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(embedder_name)

    config = PretrainedConfig()
    config.pretrained_model_name_or_path = embedder_name
    cross_attention_model = CrossAttentionModel(config)

    train_loader = create_data_loader(dataset['train'], tokenizer, batch_size)
    val_loader = create_data_loader(dataset['validation'], tokenizer, batch_size)
    test_loader = create_data_loader(dataset['test'], tokenizer, batch_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(cross_attention_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    cross_attention_model.train()
    cross_attention_model.to(device)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    best_model = None
    best_val_mae = None
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (inputs_1, inputs_2, inputs_3, label) in enumerate(train_loader):
            inputs_1 = {key: value.to(device) for key, value in inputs_1.items()}
            inputs_2 = {key: value.to(device) for key, value in inputs_2.items()}
            inputs_3 = {key: value.to(device) for key, value in inputs_3.items()}
            label = label.to(device)

            optimizer.zero_grad()
            logits = cross_attention_model(input_ids_1=inputs_1['input_ids'], attention_mask_1=inputs_1['attention_mask'],
                                        input_ids_2=inputs_2['input_ids'], attention_mask_2=inputs_2['attention_mask'],
                                        input_ids_3=inputs_3['input_ids'], attention_mask_3=inputs_3['attention_mask'])
            loss = criterion(logits, label)
            loss.backward()

            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        scheduler.step()  # Update the learning rate

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}")

        # Evaluate on validation set
        val_mae, val_loss = evaluate_model(cross_attention_model, val_loader, criterion, device)
        print(f"Validation MSE: {val_mae:.4f}, Validation Loss: {val_loss:.4f}")

        # Save the best model based on validation mse
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = cross_attention_model.state_dict()

            timestamp = int(time.time())
            checkpoint_path = f'checkpoints/best_model_{timestamp}'
            cross_attention_model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)  # Save the tokenizer

        torch.cuda.empty_cache()

    # Load the best model
    best_model = CrossAttentionModel.from_pretrained(checkpoint_path)
    best_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Evaluate on the test set
    test_mse, test_loss = evaluate_model(best_model, test_loader, criterion, device)
    print(f"Test MSE: {test_mse:.4f}, Test Loss: {test_loss:.4f}")

    cross_attention_model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
                                          

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--embedder_name', type=str, default='roberta-base')
    parser.add_argument('--dataset_path', type=str)
    args = parser.parse_args()

    # Where the final model will be saved
    model_save_path = os.environ['SM_MODEL_DIR']

    # Where the training dataset will be
    dataset_path = os.environ['SM_CHANNEL_DATASET']

    # Update the training script to use the command-line arguments
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    step_size = args.step_size
    gamma = args.gamma
    dataset_path = args.dataset_path
    accumulation_steps = args.accumulation_steps
    embedder_name = args.embedder_name

    train(num_epochs, batch_size, learning_rate, step_size, gamma, embedder_name, dataset_path, model_save_path, accumulation_steps, train_sample_size=None, val_sample_size=None, test_sample_size=None)
