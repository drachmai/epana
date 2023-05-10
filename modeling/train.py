import argparse
import os
import time
import random

import torch
from transformers import AutoTokenizer, PretrainedConfig
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import wandb

from model import ConcernDataset, CrossAttentionModel, CrossAttentionConfig


def sample_dataset(data, sample_size):
    if not sample_size or sample_size >= len(data):
        return data
    
    sampled_data = random.sample(data, sample_size)
    return sampled_data


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


def train(num_epochs, batch_size, learning_rate, step_size, gamma, embedder_name, dataset, accumulation_steps, train_sample_size=None, val_sample_size=None, test_sample_size=None):
    # Set up wandb
    wandb.init(
        project="epana",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "step_size": step_size,
            "gamma": gamma,
            "embedder_name": embedder_name,
        }
    )

    # Run on gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Place to save the checkpoints
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # Load this here and pass to model so that we can ensure there is no tokenizer mismatch later (should be same as embedder)
    tokenizer = AutoTokenizer.from_pretrained(embedder_name)

    config = CrossAttentionConfig(embedder_name)
    cross_attention_model = CrossAttentionModel(config)

    # Data loaders
    train_loader = create_data_loader(sample_dataset(dataset['train'], train_sample_size), tokenizer, batch_size)
    val_loader = create_data_loader(sample_dataset(dataset['validation'], val_sample_size), tokenizer, batch_size)
    test_loader = create_data_loader(sample_dataset(dataset['test'], test_sample_size), tokenizer, batch_size)

    # Set scheduler
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(cross_attention_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Set train and load to device (gpu if available)
    cross_attention_model.train()
    cross_attention_model.to(device)

    # Placeholders for best model tracking
    best_model = None
    best_val_mae = None

    # Training loop
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
            wandb.log({"train_loss": loss})
            loss.backward()
            epoch_loss += loss.item()

            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        wandb.log({f"epoch_{epoch}_train_loss": epoch_loss})

        # Update the learning rate
        scheduler.step()

        # Evaluate on validation set
        val_mae, val_loss = evaluate_model(cross_attention_model, val_loader, criterion, device)

        # Log val mae / loss
        wandb.log({"val_mae": val_mae})
        wandb.log({"val_loss": val_loss})

        # Save the best model based on validation mse
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = cross_attention_model.state_dict()

            timestamp = int(time.time())
            checkpoint_path = f'checkpoints/best_model_{timestamp}'

            cross_attention_model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

        torch.cuda.empty_cache()

    # Load the best model
    best_model = CrossAttentionModel.from_pretrained(checkpoint_path)
    best_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Evaluate on the test set
    test_mae, test_loss = evaluate_model(best_model, test_loader, criterion, device)

    # Log test mae / loss
    wandb.log({"test_mae": test_mae})
    wandb.log({"test_loss": test_loss})

    # Save model to wandb
    cross_attention_model.save_pretrained(wandb.run.dir)

    # Returning model and tokenizer to make training loop more generic
    return cross_attention_model, tokenizer
