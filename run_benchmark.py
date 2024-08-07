from src.ESM1bEncoder import ESM1bEncoder
from src.Esm2Encoder import Esm2Encoder
from src.MSA1bEncoder import MSA1bEncoder
from src.ProtGPTEncoder import ProtGPTEncoder
from src.ProtT5Encoder import ProtT5Encoder
from src.DataModule.ProteinDataSet import ProteinDataSet
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
import os
from src.utils.common import read_seq_from_fasta

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)


def main(args):
    connections.connect("default", host="localhost", port="19530")

    pkl_path = args.input

    with open(pkl_path, "rb") as file:
        data = pickle.load(file)

    entrys = data['entry']
    seqs = data['seqs']
    embeddings = data['embeddings']
    ecnumbers = data['ecnumbers']

    dim = embeddings[0].shape[0]

    fields = [
        FieldSchema(name="name", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=30),
        FieldSchema(name="sequence", dtype=DataType.VARCHAR, max_length=35220),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="ecnumber", dtype=DataType.VARCHAR, max_length=100)
    ]

    schema = CollectionSchema(fields,"test model contains SwissProt's embeddings.")
    db = Collection('test', schema, consistency_level="Strong")

    chunk = 20
    batch = int(len(entrys)/chunk)

    for i in range(chunk):
        entities = [
            entrys[i*batch:(i+1)*batch],
            seqs[i*batch:(i+1)*batch],
            [t.tolist() for t in embeddings[i*batch:(i+1)*batch]],
            ecnumbers[i*batch:(i+1)*batch]
        ]

        db.insert(entities)
        db.flush()

    entities = [
        entrys[(i + 1) * batch:],
        seqs[(i + 1) * batch:],
        [t.tolist() for t in embeddings[(i + 1) * batch:]],
        ecnumbers[(i + 1) * batch:]
    ]

    db.insert(entities)
    db.flush()

    '''index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }'''

    index = {
        "index_type": "FLAT",
        "metric_type": 'L2',
    }

    db.create_index("embedding", index)

    root_path = '../benchmark'
    subdirectories = os.listdir(root_path)

    for similarity_standard in ['L2', 'COS', 'IP']:

        db.load()

        for entry in tqdm(subdirectories, desc='Processing'):

            result_root_path = root_path + '/' + entry + '/' + 'test_model' + '_' + similarity_standard
            os.makedirs(result_root_path, exist_ok=True)

            pooler_token = embeddings[entrys.index(entry)].tolist()
            # print(pooler_token)

            if similarity_standard == 'L2':
                index_standart = 'L2'

            elif similarity_standard == 'COS':
                index_standart = 'COSINE'

                db.release()
                db.drop_index()

                '''index = {
                    "index_type": "IVF_FLAT",
                    "metric_type": 'COSINE',
                    "params": {"nlist": 128},
                }'''

                index = {
                    "index_type": "FLAT",
                    "metric_type": 'COSINE',
                }

                db.create_index("embedding", index)

                db.load()

            elif similarity_standard == 'IP':
                index_standart = 'IP'

                db.release()
                db.drop_index()

                '''index = {
                    "index_type": "IVF_FLAT",
                    "metric_type": 'IP',
                    "params": {"nlist": 128},
                }'''

                index = {
                    "index_type": "FLAT",
                    "metric_type": 'IP',
                }

                db.create_index("embedding", index)

                db.load()

            search_params = {
                "metric_type": index_standart,
                "params": {"nprobe": 10},
            }

            result = db.search([pooler_token], "embedding", search_params, limit=250, output_fields=["ecnumber"])

            name = []
            ecnumber = []
            valus = []

            for hits in result:
                for hit in hits:
                    name.append(hit.id)
                    valus.append(hit.distance)
                    ecnumber.append(hit.entity.get('ecnumber'))

            df = pd.DataFrame({'Entry': name, 'Score': valus, 'ECnumber': ecnumber})

            result_path = result_root_path + '/result.tsv'
            df.to_csv(result_path, sep='\t')

        db.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input file path and Output file path.')

    parser.add_argument('-i', '--input', type=str, help='Input pkl file path.')

    args = parser.parse_args()
    main(args)
