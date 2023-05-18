# tests/test_model.py

import unittest
import torch
from transformers import AutoTokenizer
from epana_modeling.model import CrossAttentionModel, CrossAttentionConfig

class TestMyModel(unittest.TestCase):

    def test_cross_attention_model_output(self):
        embedder_name = "distilroberta-base"
        config = CrossAttentionConfig(embedder_name)
        model = CrossAttentionModel(config)
        tokenizer = AutoTokenizer.from_pretrained(embedder_name)

        input_1 = tokenizer("This is a sample text 1", return_tensors="pt", padding=True)
        input_2 = tokenizer("This is a sample text 2", return_tensors="pt", padding=True)
        input_3 = tokenizer("This is a sample text 3", return_tensors="pt", padding=True)

        logits = model(input_ids_1=input_1['input_ids'], attention_mask_1=input_1['attention_mask'],
            input_ids_2=input_2['input_ids'], attention_mask_2=input_2['attention_mask'],
            input_ids_3=input_3['input_ids'], attention_mask_3=input_3['attention_mask'])
        
        self.assertEqual(logits.shape, torch.Size([]))


if __name__ == "__main__":
    unittest.main()