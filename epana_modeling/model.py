import torch
from torch.utils.data import Dataset
from transformers import AutoModel, PretrainedConfig, PreTrainedModel


class CrossAttentionConfig(PretrainedConfig):
    model_type = "cross_attention"
    def __init__(self, pretrained_model_name_or_path: str = None, **kwargs):
        super(CrossAttentionConfig, self).__init__(**kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path


class ConcernDatasetBatchSampler:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class ConcernDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data, self.sorted_lengths = self._sort_by_length(data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        inputs_1 = row['previous_chat']
        inputs_2 = row['last_message']
        inputs_3 = row['concerning_definitions']
        label = torch.tensor(row['is_concerning_score']).float()

        return {"previous_chat": inputs_1, "last_message": inputs_2, "concerning_definitions": inputs_3, "label": label}
    
    def _sort_by_length(self, data):
        # Calculate the length of each text input
        input_lengths = [max(len(self.tokenizer.encode(row['previous_chat'], truncation=True)),
                             len(self.tokenizer.encode(row['last_message'], truncation=True)),
                             len(self.tokenizer.encode(row['concerning_definitions'], truncation=True)))
                        for row in data]

        # Sort the data and lengths by input length
        sorted_indices = sorted(range(len(input_lengths)), key=lambda i: input_lengths[i])
        sorted_data = [data[i] for i in sorted_indices]
        sorted_lengths = [input_lengths[i] for i in sorted_indices]

        return sorted_data, sorted_lengths


class CrossAttentionModel(PreTrainedModel):
    config_class = CrossAttentionConfig
    
    def __init__(self, config, tokenizer=None):
        super(CrossAttentionModel, self).__init__(config)
        if config.pretrained_model_name_or_path:
            self.embedding_model = AutoModel.from_pretrained(config.pretrained_model_name_or_path)
        else:
            self.embedding_model = None

        self.tokenizer = tokenizer
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=768, num_heads=12)
        self.classifier = torch.nn.Linear(768, 1)

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(768, 3072),
            torch.nn.ReLU(),
            torch.nn.Linear(3072, 768),
        )

        self.layer_norm = torch.nn.LayerNorm(768)

    def preprocess_input(self, text, max_length=None):
        input_dict = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length=max_length, truncation=True)
        return input_dict['input_ids'], input_dict['attention_mask']

    def forward(self, previous_chat, last_message, concerning_definitions):
        max_length = max(len(self.tokenizer(previous_chat)['input_ids'][0]), len(self.tokenizer(last_message)['input_ids'][0]), len(self.tokenizer(concerning_definitions)['input_ids'][0]))

        input_ids_1, attention_mask_1 = self.preprocess_input(previous_chat, max_length=max_length)
        input_ids_2, attention_mask_2 = self.preprocess_input(last_message, max_length=max_length)
        input_ids_3, attention_mask_3 = self.preprocess_input(concerning_definitions, max_length=max_length)

        # Remove the extra dimension
        input_ids_1 = input_ids_1.squeeze(1)
        attention_mask_1 = attention_mask_1.squeeze(1)
        input_ids_2 = input_ids_2.squeeze(1)
        attention_mask_2 = attention_mask_2.squeeze(1)
        input_ids_3 = input_ids_3.squeeze(1)
        attention_mask_3 = attention_mask_3.squeeze(1)

        # Get embeddings embedding model
        outputs_1 = self.embedding_model(input_ids=input_ids_1, attention_mask=attention_mask_1)
        outputs_2 = self.embedding_model(input_ids=input_ids_2, attention_mask=attention_mask_2)
        outputs_3 = self.embedding_model(input_ids=input_ids_3, attention_mask=attention_mask_3)

        # Apply cross-attention
        cross_attention_output, _ = self.cross_attention(query=outputs_1.last_hidden_state.permute(1, 0, 2),
                                                         key=outputs_2.last_hidden_state.permute(1, 0, 2),
                                                         value=outputs_3.last_hidden_state.permute(1, 0, 2),
                                                         attn_mask=None)
        
        # Normalization
        cross_attention_output = self.layer_norm(cross_attention_output)

        # FFN
        cross_attention_output = self.ffn(cross_attention_output)

        # Mean pooling
        pooled_output = torch.mean(cross_attention_output, dim=0)
        logits = self.classifier(pooled_output).squeeze()
        
        return logits
