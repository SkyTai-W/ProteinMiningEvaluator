import pandas as pd
import torch
import torch.nn as nn
from src.Model.ProteinEncoder import ProteinEncoder
from torch.cuda.amp import autocast, GradScaler


class ProtGPTEncoder(ProteinEncoder):
    def __init__(self, model_name='ProtGPT'):
        super(ProtGPTEncoder, self).__init__(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_batch(self, batch):
        names = batch['names']
        seqs = batch['seqs']
        ecnumbers = batch['ECnumber']

        scaler = GradScaler()
        with autocast():
            token = self.tokenizer(seqs, return_tensors="pt", truncation=True, max_length=1024, padding=True,
                                   add_special_tokens=True).to(self.device)

            input_ids = token['input_ids']
            attention_mask = token['attention_mask']

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']

        output = output.float()

        pooler_token = torch.mean(output, dim=1)

        return {'names': names, 'seqs': seqs, 'embeddings': pooler_token, 'ecnumbers': ecnumbers}

    def forward(self, input):
        scaler = GradScaler()
        with autocast():
            token = self.tokenizer(input, return_tensors="pt", truncation=False, padding=True,
                                   add_special_tokens=True).to(self.device)

            input_ids = token['input_ids']
            attention_mask = token['attention_mask']

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']

        output = output.float()

        self.pooler_token = torch.mean(output, dim=1)

        return self.pooler_token
