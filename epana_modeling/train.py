import os
import time
import random
from functools import partial

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import wandb
from tqdm import tqdm
from torch.utils.data import BatchSampler, RandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from accelerate import Accelerator

from model import ConcernDataset, ConcernModel, ConcernModelConfig, BinaryFocalLoss


def sample_dataset(data, sample_size):
    if not sample_size or sample_size >= len(data):
        return data
    
    sampled_data = random.sample(data, sample_size)
    return sampled_data


def get_classification_metrics(logits, targets):
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()

    # Convert targets to cpu
    targets = targets.cpu().numpy()

    # Get class predictions
    preds = np.argmax(probs, axis=1)

    # Probability of the positive class
    probs = probs[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(targets, preds)
    auc = roc_auc_score(targets, probs)
    f1 = f1_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)

    return accuracy, auc, f1, precision, recall


def evaluate_model(model, data_loader, criterion, mode, device):
    model.eval()
    total_loss = 0
    mae_sum = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            previous_chat_input_ids = batch["previous_chat"]["input_ids"].to(device)
            previous_chat_attention_mask = batch["previous_chat"]["attention_mask"].to(device)

            last_message_input_ids = batch["last_message"]["input_ids"].to(device)
            last_message_attention_mask = batch["last_message"]["attention_mask"].to(device)

            concerning_definitions_input_ids = batch["concerning_definitions"]["input_ids"].to(device)
            concerning_definitions_attention_mask = batch["concerning_definitions"]["attention_mask"].to(device)

            target = batch["target"].to(device)

            logits = model(
                previous_chat={
                    'input_ids': previous_chat_input_ids, 
                    'attention_mask': previous_chat_attention_mask
                },
                last_message={
                    'input_ids': last_message_input_ids, 
                    'attention_mask': last_message_attention_mask
                },
                concerning_definitions={
                    'input_ids': concerning_definitions_input_ids, 
                    'attention_mask': concerning_definitions_attention_mask
                }
            )

            loss = criterion(logits, target)
            total_loss += loss.item()

            if mode == "regression":
                # Calculate mean absolute error (MAE)
                mae_sum += torch.abs(logits - target).sum().item()
            elif mode == "classification":
                local_targets = target.cpu().numpy()
                
                probs = torch.sigmoid(logits).cpu().detach().numpy().flatten()
                preds = (probs > 0.5).astype(int)


                all_preds.extend(preds)
                all_targets.extend(local_targets)
                all_probs.extend(probs)

    if mode == "regression":
        avg_mae = mae_sum / len(data_loader.dataset)
        metrics = {
            "avg_mae": avg_mae,
            "avg_loss": total_loss / len(data_loader)
        }
        tracked_metric = avg_mae
    elif mode == "classification":
        auc = roc_auc_score(all_targets, all_probs)
        metrics = {
            "accuracy": accuracy_score(all_targets, all_preds),
            "auc": auc,
            "f1": f1_score(all_targets, all_preds),
            "precision": precision_score(all_targets, all_preds),
            "recall": recall_score(all_targets, all_preds)
        }   
        tracked_metric = auc * -1

    metrics['loss'] = total_loss / len(data_loader)
    
    return tracked_metric, metrics


def train(num_epochs, batch_size, learning_rate, step_size, gamma, embedder_name, dataset, accumulation_steps, strategy, mode, focal_alpha=None, focal_gamma=None, train_sample_size=None, val_sample_size=None, test_sample_size=None):
    # Start accelerator
    accelerator = Accelerator()
    
    # Set up wandb
    log_config = {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "step_size": step_size,
            "gamma": gamma,
            "embedder_name": embedder_name,
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            "strategy": strategy,
            "mode": mode
        }
    wandb.init(
        project="epana",
        config=log_config
    )

    # Run on gpu is available
    device = accelerator.device

    # Place to save the checkpoints
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # Load this here and pass to model so that we can ensure there is no tokenizer mismatch later (should be same as embedder)
    tokenizer = AutoTokenizer.from_pretrained(embedder_name)

    config = ConcernModelConfig(embedder_name, strategy=strategy)
    
    concern_model = ConcernModel(config)

    # Set train and load to device (gpu if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    concern_model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        concern_model = torch.nn.DataParallel(concern_model)

    # Data loaders
    train_dataset = ConcernDataset(sample_dataset(dataset['train'], train_sample_size), tokenizer, mode)
    validation_dataset = ConcernDataset(sample_dataset(dataset['validation'], val_sample_size), tokenizer, mode)
    test_dataset = ConcernDataset(sample_dataset(dataset['test'], test_sample_size), tokenizer, mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Set scheduler
    if mode == "regression":
        criterion = torch.nn.MSELoss()
    elif mode == "classification":
        criterion = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    criterion = criterion.to(device)  # Move the criterion to the device

    optimizer = torch.optim.Adam(concern_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    concern_model.train()

    # Placeholders for best model tracking
    best_model = None

    best_val_metric = float('inf')

    # Prepare with accelerator
    train_loader, validation_loader, test_loader, concern_model, optimizer, criterion = accelerator.prepare(
        train_loader, validation_loader, test_loader, concern_model, optimizer, criterion
    )

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):            
            previous_chat_input_ids = batch["previous_chat"]["input_ids"].to(device)
            previous_chat_attention_mask = batch["previous_chat"]["attention_mask"].to(device)

            last_message_input_ids = batch["last_message"]["input_ids"].to(device)
            last_message_attention_mask = batch["last_message"]["attention_mask"].to(device)

            concerning_definitions_input_ids = batch["concerning_definitions"]["input_ids"].to(device)
            concerning_definitions_attention_mask = batch["concerning_definitions"]["attention_mask"].to(device)

            target = batch["target"].to(device)

            optimizer.zero_grad()
            with accelerator.autocast():
                logits = concern_model(
                    previous_chat={
                        'input_ids': previous_chat_input_ids, 
                        'attention_mask': previous_chat_attention_mask
                    },
                    last_message={
                        'input_ids': last_message_input_ids, 
                        'attention_mask': last_message_attention_mask
                    },
                    concerning_definitions={
                        'input_ids': concerning_definitions_input_ids, 
                        'attention_mask': concerning_definitions_attention_mask
                    }
                )
                loss = criterion(logits, target)
            accelerator.backward(loss)

            epoch_loss += accelerator.gather(loss).item()

            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        wandb.log({f"train_loss": epoch_loss})

        # Update the learning rate
        wandb.log({"learning_rate": scheduler.get_lr()[0]})
        scheduler.step()

        # Evaluate on validation set
        tracked_val_metric, val_metrics = evaluate_model(concern_model, validation_loader, criterion, mode, device)

        # Log val mae / loss
        wandb.log({"val_metrics": val_metrics})
        for name, param in concern_model.named_parameters():
            if param.grad is not None:
                wandb.log({f'{name}.grad': wandb.Histogram(param.grad.cpu().numpy())})


        # Save the best model based on validation mse
        if tracked_val_metric < best_val_metric:
            best_val_metric = tracked_val_metric
            best_model = concern_model.state_dict()

            timestamp = int(time.time())
            checkpoint_path = f'checkpoints/best_model_{timestamp}'

            concern_model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

        torch.cuda.empty_cache()

    # Load the best model
    best_model = ConcernModel.from_pretrained(checkpoint_path)
    best_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Evaluate on the test set
    _, test_metrics = evaluate_model(best_model, test_loader, criterion, mode, device)

    # Log test mae / loss
    wandb.log({"test_metrics": test_metrics})

    # Save model to wandb
    best_model.save_pretrained(wandb.run.dir)

    # Returning model and tokenizer to make training loop more generic
    return best_model, tokenizer
