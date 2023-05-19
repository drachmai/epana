# tests/test_model.py

import unittest
import torch
from transformers import AutoTokenizer
from epana_modeling.model import ConcernModel, ConcernModelConfig, ConcernDataset, BinaryFocalLoss

class TestConcernModel(unittest.TestCase):

    def test_model_output(self):
        embedder_name = "distilroberta-base"
        config = ConcernModelConfig(embedder_name)
        model = ConcernModel(config)
        tokenizer = AutoTokenizer.from_pretrained(embedder_name)

        previous_chat = "Human - I am thinking of hurting myself"
        last_message = "Assistant - Suicide is a good way to make sure you don't have a bad day."
        concerning_definitions = "Discouraging self harm"

        previous_chat_tokenized = tokenizer(previous_chat, padding=True, truncation=True, return_tensors="pt")
        last_message_tokenized = tokenizer(last_message, padding=True, truncation=True, return_tensors="pt")
        concerning_definitions_tokenized = tokenizer(concerning_definitions, padding=True, truncation=True, return_tensors="pt")

        logits = model(previous_chat_tokenized, last_message_tokenized, concerning_definitions_tokenized)
        
        self.assertEqual(logits.shape, torch.Size([1, 1]))

    def test_binary_focal_loss(self):
        model = torch.nn.Linear(100, 1)
        criterion = BinaryFocalLoss()
        input = torch.randn(128, 100)
        target = torch.empty(128, dtype=torch.float32).random_(2)
        
        output = model(input)
        loss = criterion(output, target)
        
        assert loss.item() > 0

if __name__ == "__main__":
    unittest.main()