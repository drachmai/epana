import json
import argparse
import os

import torch
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import pandas as pd

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


def train(paths, train_sample_size=None, val_sample_size=None, test_sample_size=None):

    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    embedding_model = RobertaModel.from_pretrained('roberta-base')  
    cross_attention_model = CrossAttentionModel(embedding_model)

    train_data, val_data, test_data = process_datasets(paths, train_sample_size, val_sample_size, test_sample_size)

    train_loader = create_data_loader(train_data, tokenizer, batch_size)
    val_loader = create_data_loader(val_data, tokenizer, batch_size)
    test_loader = create_data_loader(test_data, tokenizer, batch_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(cross_attention_model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    accumulation_steps = 4  # Adjust this value according to your needs

    cross_attention_model.train()
    cross_attention_model.to(device)

    num_epochs = 5
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
        if not best_val_mae:
            best_val_mae = val_mae
            torch.save(cross_attention_model.state_dict(), 'best_model.pth')
        elif val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(cross_attention_model.state_dict(), 'best_model.pth')

    torch.cuda.empty_cache()

    # Load the best model
    best_model = CrossAttentionModel(model)
    best_model.load_state_dict(torch.load('best_model.pth'))
    best_model.to(device)

    # Evaluate on the test set
    test_mse, test_loss = evaluate_model(best_model, test_loader, criterion, device)
    print(f"Test MSE: {test_mse:.4f}, Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('input_files', metavar='input_file', type=str, nargs='+',
                        help='List of input file paths')
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

    train(input_paths, train_sample_size, val_sample_size, test_sample_size)