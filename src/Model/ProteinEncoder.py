import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel as parallel
from transformers import EsmModel, EsmTokenizer, GPT2Model, GPT2Tokenizer
from transformers import T5EncoderModel, T5Tokenizer
from abc import ABC, abstractmethod
import esm


class ProteinEncoder(nn.Module):
    def __init__(self, model_name='ESM2_650M'):
        super(ProteinEncoder, self).__init__()
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.model_name in ['ESM2_650M', 'Esm2_650M', 'esm2_650M', 'ESM2_650', 'Esm2_650',
                               'esm2_650', 'esm2_t33_650M_UR50D']:
            self.model = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D').to(self.device)
            self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')

        elif self.model_name in ['ESM2_3B', 'Esm2_3B', 'esm2_3B', 'ESM2_3', 'Esm2_3',
                               'esm2_3', 'esm2_t36_3B_UR50D']:
            self.model = EsmModel.from_pretrained('facebook/esm2_t36_3B_UR50D').to(self.device)
            self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D')

        elif self.model_name in ['ProtT5', 'T5', 'protT5', 'prott5', 't5']:
            self.model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50').to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')

        elif self.model_name in ['ProtGPT', 'protGPT', 'Protgpt', 'protGpt', 'protgpt', 'GPT', 'gpt', 'Gpt']:
            self.model = GPT2Model.from_pretrained('nferruz/ProtGPT2').to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained('nferruz/ProtGPT2')

        elif self.model_name in ['MSA1b', 'Msa1b', 'msa1b', 'ESM_MSA1b', 'ESM_Msa1b', 'ESM_msa1b', 'Esm_MSA1b',
                                 'Esm_Msa1b', 'Esm_msa1b', 'esm_MSA1b', 'esm_Msa1b', 'esm_msa1b']:
            self.model,alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
            self.model.to(self.device)
            self.tokenizer = alphabet.get_batch_converter()

        elif self.model_name in ['ESM1b','Esm1b','esm1b','ESM1b_t33_650M_UR50S','Esm1b_t33_650M_UR50S','esm1b_t33_650M_UR50S']:
            self.model = EsmModel.from_pretrained('facebook/esm1b_t33_650M_UR50S').to(self.device)
            self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm1b_t33_650M_UR50S')


        else:
            raise ValueError('Unsupported model name: {}'.format(self.model_name))

        self.model.eval()

    @abstractmethod
    def encode_batch(self, batch):
        pass

    @abstractmethod
    def forward(self, input):
        pass

    def __call__(self, input):
        return self.forward(input)
