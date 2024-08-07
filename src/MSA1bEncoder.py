import torch
import torch.nn as nn
from src.Model.ProteinEncoder import ProteinEncoder
from torch.cuda.amp import autocast, GradScaler
import esm


class MSA1bEncoder(ProteinEncoder):
    def __init__(self, model_name='MSA1b'):
        super(MSA1bEncoder, self).__init__(model_name)

    def encode_batch(self, batch):
        names = batch['names']
        seqs = batch['seqs']

        scaler = GradScaler()
        max_length = max(len(seq) for seq in seqs)  # 找到最长序列的长度
        token_len = [len(seq) for seq in seqs]

        # 对序列进行填充
        padded_seqs = [seq + ' ' * (max_length - len(seq)) for seq in seqs]

        with autocast():
            label, str, token = self.tokenizer([(name, seq) for name, seq in zip(names, padded_seqs)])
            tmp = token.to(self.device)

            with torch.no_grad():
                output = self.model(tmp, repr_layers=[12], return_contacts=True)["representations"][12][0]

        output = output.float()

        self.pooler_token = []
        for i, seq_len in enumerate(token_len):
            self.pooler_token.append(output[i, 1:seq_len - 1].mean(0))

        return {'names': names, 'seqs': seqs, 'embeddings': self.pooler_token}

    def forward(self, input):
        if len(input)>1022:
            input = input[:1022]

        scaler = GradScaler()
        with autocast():
            label, str, token = self.tokenizer([('Protein', input)])
            tmp = token.to(self.device)

            with torch.no_grad():
                output = self.model(tmp, repr_layers=[12], return_contacts=True)["representations"][12][0]

        token_len = len(output[0])

        pooler_token = torch.mean(output[0, 1:token_len], dim=0)

        return pooler_token


def main():
    pass


if __name__ == '__main__':
    main()
