import torch
from torch.utils.data import Dataset

class ConcernDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs_1 = self._prepare_inputs(row['previous_chat'])
        inputs_2 = self._prepare_inputs(row['last_message'])
        inputs_3 = self._prepare_inputs(row['concerning_definitions'])
        label = torch.tensor(row['is_concerning_score']).float()

        return (inputs_1, inputs_2, inputs_3, label)
    
    def _prepare_inputs(self, text):
        tokens = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        return tokens


class CrossAttentionModel(torch.nn.Module):
    def __init__(self, embedding_model):
        super(CrossAttentionModel, self).__init__()
        self.embedding_model = embedding_model
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=768, num_heads=12)
        self.classifier = torch.nn.Linear(768, 1)

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(768, 3072),
            torch.nn.ReLU(),
            torch.nn.Linear(3072, 768),
        )

        self.layer_norm = torch.nn.LayerNorm(768)

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, input_ids_3, attention_mask_3):
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
        
        #Normalization (play with this)
        cross_attention_output = self.layer_norm(cross_attention_output)

        # FFN (play with this)
        cross_attention_output = self.ffn(cross_attention_output)

        # Mean pooling (play with this, only this or max pooling should be on)
        pooled_output = torch.mean(cross_attention_output, dim=0)
        logits = self.classifier(pooled_output).squeeze()

        # Max pooling (play with this, only this or mean pooling should be on)
        #pooled_output, _ = torch.max(cross_attention_output, dim=0)
        #logits = self.classifier(pooled_output).squeeze()
        
        # Use the first token's representation for classification (CLS token)
        # logits = self.classifier(cross_attention_output[0]).squeeze()
        
        return logits
