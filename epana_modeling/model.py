import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=[1, 1], gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        # Move alpha to the correct device
        alpha = self.alpha.to(inputs.device)
        
        BCE_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = alpha[targets] * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class ConcernDataset(Dataset):
    def __init__(self, dataset, tokenizer, mode="regression"):
        previous_chats = dataset['previous_chat']
        last_messages = dataset['last_message']
        concerning_definitions = dataset['concerning_definitions']

        if mode == "classification":
            target = torch.tensor((np.array(dataset['is_concerning_score']) >= 0).astype(int)).long()
        elif mode == "regression":
            target = torch.tensor(dataset['is_concerning_score']).float()

        # Tokenize the data
        tokenized_previous_chats = tokenizer(previous_chats, padding=True, truncation=True, return_tensors="pt")
        tokenized_last_messages = tokenizer(last_messages, padding=True, truncation=True, return_tensors="pt")
        tokenized_concerning_definitions = tokenizer(concerning_definitions, padding=True, truncation=True, return_tensors="pt")

        self.data = {
            'previous_chat': {
                'input_ids': tokenized_previous_chats['input_ids'],
                'attention_mask': tokenized_previous_chats['attention_mask']
            },
            'last_message': {
                'input_ids': tokenized_last_messages['input_ids'],
                'attention_mask': tokenized_last_messages['attention_mask']
            },
            'concerning_definitions': {
                'input_ids': tokenized_concerning_definitions['input_ids'],
                'attention_mask': tokenized_concerning_definitions['attention_mask']
            },
            'target': target
        }

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, idx):
        return {
            'previous_chat': {"input_ids": self.data['previous_chat']["input_ids"][idx], "attention_mask": self.data["previous_chat"]["attention_mask"][idx]},
            'last_message': {"input_ids": self.data['last_message']["input_ids"][idx], "attention_mask": self.data["last_message"]["attention_mask"][idx]},
            'concerning_definitions': {"input_ids": self.data['concerning_definitions']["input_ids"][idx], "attention_mask": self.data["concerning_definitions"]["attention_mask"][idx]},
            'target': self.data['target'][idx]
        }


class ConcernModelConfig(PretrainedConfig):
    def __init__(self, pretrained_model_name_or_path=None, strategy="concatenate", mode="classification", **kwargs):
        super(ConcernModelConfig, self).__init__(**kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.strategy = strategy
        self.supported_strategies = ["concatenate", "average", "attention", "higher_level_activations"]
        self.mode = mode


class ConcernModel(PreTrainedModel):
    config_class = ConcernModelConfig

    def __init__(self, config):
        super(ConcernModel, self).__init__(config)
        if config.pretrained_model_name_or_path:
            self.embedding_model = AutoModel.from_pretrained(config.pretrained_model_name_or_path)
        else:
            self.embedding_model = None
            self.embedding_cache = {}

        self.mode = config.mode

        if config.mode == "classification":
            classifier_output_size = 2
        elif config.mode == "regression":
            classifier_output_size = 1

        if config.strategy == 'concatenate':
            self.combine_embeddings = self.concatenate_embeddings
            self.classifier = torch.nn.Linear(3 * 768, classifier_output_size)
        elif config.strategy == 'average':
            self.combine_embeddings = self.average_embeddings
            self.classifier = torch.nn.Linear(768, classifier_output_size)
        elif config.strategy == 'attention':
            self.combine_embeddings = self.attention_embeddings
            self.attention_layer = torch.nn.MultiheadAttention(embed_dim=768, num_heads=12)
            self.classifier = torch.nn.Linear(768, classifier_output_size)
        elif config.strategy == 'higher_level_activations':
            self.combine_embeddings = self.higher_level_activations
            self.classifier = nn.Linear(768, classifier_output_size)
        else:
            raise ValueError(f"Invalid strategy '{config.strategy}'. Supported strategies: {config.supported_strategies}")

    def forward(self, previous_chat, last_message, concerning_definitions):
        previous_chat_outputs = self.embedding_model(**previous_chat)
        last_message_outputs = self.embedding_model(**last_message)
        concerning_definitions_outputs = self.embedding_model(**concerning_definitions)

        combined_embeddings = self.combine_embeddings(previous_chat_outputs, last_message_outputs, concerning_definitions_outputs)

        logits = self.classifier(combined_embeddings)
        
        if self.mode == "regression":
            logits = logits.squeeze()

        return logits

    def concatenate_embeddings(self, emb1, emb2, emb3):
        emb1_cls = emb1.last_hidden_state[:, 0, :]
        emb2_cls = emb2.last_hidden_state[:, 0, :]
        emb3_cls = emb3.last_hidden_state[:, 0, :]
        return torch.cat((emb1_cls, emb2_cls, emb3_cls), dim=1)

    def average_embeddings(self, emb1, emb2, emb3):
        emb1_cls = emb1.last_hidden_state[:, 0, :]
        emb2_cls = emb2.last_hidden_state[:, 0, :]
        emb3_cls = emb3.last_hidden_state[:, 0, :]
        return (emb1_cls + emb2_cls + emb3_cls) / 3

    def attention_embeddings(self, emb1, emb2, emb3):
        emb1_cls = emb1.last_hidden_state[:, 0, :].unsqueeze(0)
        emb2_cls = emb2.last_hidden_state[:, 0, :].unsqueeze(0)
        emb3_cls = emb3.last_hidden_state[:, 0, :].unsqueeze(0)

        query = emb1_cls.permute(1, 0, 2)
        key = emb2_cls.permute(1, 0, 2)
        value = emb3_cls.permute(1, 0, 2)

        attention_output, _ = self.attention_layer(query=query, key=key, value=value)
        return attention_output.squeeze(0)
    
    def higher_level_activations(self, emb1, emb2, emb3):
        emb1_activations = emb1.last_hidden_state[:, -4:, :].mean(dim=1)
        emb2_activations = emb2.last_hidden_state[:, -4:, :].mean(dim=1)
        emb3_activations = emb3.last_hidden_state[:, -4:, :].mean(dim=1)

        return (emb1_activations + emb2_activations + emb3_activations) / 3


