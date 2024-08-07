import pandas as pd
import torch
import torch.nn as nn
from src.Model.ProteinEncoder import ProteinEncoder
from torch.cuda.amp import autocast, GradScaler


class Esm2Encoder(ProteinEncoder):
    def __init__(self, model_name='ESM2_650M'):
        super(Esm2Encoder, self).__init__(model_name)

    def encode_batch(self, batch):
        names = batch['names']
        seqs = batch['seqs']
        ecnumbers = batch['ECnumber']

        scaler = GradScaler()
        with autocast():
            token = self.tokenizer(seqs, return_tensors="pt", truncation=True,max_length=7000, padding=True).to(
                self.device)

            batch_lens = (token['attention_mask']).sum(1)

            with torch.no_grad():
                output = self.model(**token)['last_hidden_state']

        output = output.float()
        tmp = []
        for idx, token_len in enumerate(batch_lens):
            tmp.append(torch.mean(output[idx, 1:token_len - 1], dim=0))

        pooler_token = tmp

        return {'names': names, 'seqs': seqs, 'embeddings': pooler_token,'ecnumbers':ecnumbers}

    def forward(self, input):
        scaler = GradScaler()
        with autocast():
            token = self.tokenizer(input, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

            with torch.no_grad():
                output = self.model(**token)['last_hidden_state']

        token_len = len(output[0])

        self.pooler_token = torch.mean(output[0, 1:token_len - 1], dim=0)

        return self.pooler_token


def main():
    pass


if __name__ == '__main__':
    main()
