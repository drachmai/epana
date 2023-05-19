import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
import numpy as np
        

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        inputs = inputs.squeeze(-1)
        BCE_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class ConcernDataset(Dataset):
    def __init__(self, dataset, tokenizer, mode="classification"):
        previous_chats = dataset['previous_chat']
        last_messages = dataset['last_message']
        concerning_definitions = dataset['concerning_definitions']

        if mode == "classification":
            target = torch.tensor((np.array(dataset['is_concerning_score']) >= 0).astype(int)).float()
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
    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super(ConcernModelConfig, self).__init__(**kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path


class ConcernModel(PreTrainedModel):
    config_class = ConcernModelConfig

    def __init__(self, config):
        super(ConcernModel, self).__init__(config)

        if config.pretrained_model_name_or_path:
            self.previous_chat_embedder = AutoModel.from_pretrained(config.pretrained_model_name_or_path)
            self.last_message_embedder = AutoModel.from_pretrained(config.pretrained_model_name_or_path)
            self.concerning_definition_embedder = AutoModel.from_pretrained(config.pretrained_model_name_or_path)
        else:
            self.previous_chat_embedder = None
            self.last_message_embedder = None
            self.concerning_definition_embedder = None

        self.projection = nn.Linear(self.previous_chat_embedder.config.hidden_size * 2, self.previous_chat_embedder.config.hidden_size)
        self.classifier = nn.Linear(self.previous_chat_embedder.config.hidden_size * 2, 1)

    @staticmethod
    def compute_co_attention(a, b):
        similarity_matrix = torch.bmm(a, b.transpose(1, 2))
        a_attention = torch.softmax(similarity_matrix, dim=1)
        b_attention = torch.softmax(similarity_matrix.transpose(1, 2), dim=1)
        a_attended = torch.bmm(a_attention, b)
        b_attended = torch.bmm(b_attention, a)

        # Apply mean pooling separately to a_attended and b_attended
        a_attended_pooled = a_attended.mean(dim=1)
        b_attended_pooled = b_attended.mean(dim=1)

        return torch.cat((a_attended_pooled, b_attended_pooled), dim=-1)

    def forward(self, previous_chat, last_message, concerning_definitions):
        previous_chat_outputs = self.previous_chat_embedder(**previous_chat).last_hidden_state
        last_message_outputs = self.last_message_embedder(**last_message).last_hidden_state
        concerning_definitions_outputs = self.concerning_definition_embedder(**concerning_definitions).last_hidden_state

        co_attention_chat_last_message = self.compute_co_attention(previous_chat_outputs, last_message_outputs)

        # Projecting back to the original dimension
        co_attention_chat_last_message_projected = self.projection(co_attention_chat_last_message)

        co_attention = self.compute_co_attention(co_attention_chat_last_message_projected.unsqueeze(1), concerning_definitions_outputs)

        logits = self.classifier(co_attention)

        return logits

