import pandas as pd
import torch
import torch.nn as nn
from src.Model.ProteinEncoder import ProteinEncoder
from torch.cuda.amp import autocast, GradScaler
import re


class ProtT5Encoder(ProteinEncoder):
    def __init__(self, model_name='ProtT5'):
        super(ProtT5Encoder, self).__init__(model_name)

    def encode_batch(self, batch):
        names = batch['names']
        seqs = batch['seqs']
        seqs = [seq[:1024] if len(seq) > 1024 else seq for seq in seqs]
        ecnumbers = batch['ECnumber']

        length = [len(seq) for seq in seqs]

        cut_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in seqs]

        scaler = GradScaler()
        with autocast():
            ids = self.tokenizer.batch_encode_plus(cut_seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(self.device)
            attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']

        pooling_layer = nn.AdaptiveAvgPool1d(1)

        max_length = output.size()[1]
        batch_size = len(seqs)
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool).to(self.device)

        for i, l in enumerate(length):
            mask[i, :l] = True

        # 将特征矩阵与掩码相乘，使得超出字符串长度的部分都变为零
        masked_feats = output * mask.unsqueeze(-1).float()

        # 使用池化层对特征矩阵进行池化操作
        pooler_token = pooling_layer(masked_feats.permute(0, 2, 1)).squeeze(-1)

        return {'names': names, 'seqs': seqs, 'embeddings': pooler_token, 'ecnumbers': ecnumbers}

    def forward(self, input):
        scaler = GradScaler()
        input = [input[:1024] if len(input) > 1024 else input]
        length = len(input)

        cut_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in input]

        scaler = GradScaler()
        with autocast():
            ids = self.tokenizer.batch_encode_plus(cut_seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(self.device)
            attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']

            emb_0 = output[0, :length]
            self.pooler_token = emb_0.mean(dim=0)

        return self.pooler_token
