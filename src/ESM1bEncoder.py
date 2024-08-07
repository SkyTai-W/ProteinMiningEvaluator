import torch
import torch.nn as nn
from src.Model.ProteinEncoder import ProteinEncoder
from torch.cuda.amp import autocast, GradScaler
import esm


class ESM1bEncoder(ProteinEncoder):
    def __init__(self, model_name='ESM1b'):
        super(ESM1bEncoder, self).__init__(model_name)

    def encode_batch(self, batch):
        names = batch['names']
        seqs = batch['seqs']
        ecnumbers = batch['ECnumber']

        scaler = GradScaler()

        with autocast():
            token = self.tokenizer(seqs, return_tensors="pt", truncation=True, max_length=1022, padding=True).to(
                self.device)

            batch_lens = (token['attention_mask'] != 0).sum(1)

            with torch.no_grad():
                output = self.model(**token)["last_hidden_state"]

        output = output.float()

        tmp = []
        for idx, token_len in enumerate(batch_lens):
            tmp.append(torch.mean(output[idx, 1:token_len - 1], dim=0))

        pooler_token = tmp

        return {'names': names, 'seqs': seqs, 'embeddings': pooler_token, 'ecnumbers': ecnumbers}

    def forward(self, input):
        scaler = GradScaler()
        with autocast():
            label, str, token = self.tokenizer([('Protein', input)])
            tmp = token.to(self.device)

            with torch.no_grad():
                output = self.model(tmp, repr_layers=[12], return_contacts=True)["representations"][12][0]

        token_len = len(output[0])

        self.pooler_token = torch.mean(output[0, 1:token_len], dim=0)

        return self.pooler_token


def main():
    pass


if __name__ == '__main__':
    main()
