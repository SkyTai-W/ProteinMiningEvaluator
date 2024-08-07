from torch.utils.data import Dataset
import pandas as pd


class ProteinDataSet(Dataset):
    def __init__(self, csv_path):
        if csv_path.endswith('.csv'):
            self.data = pd.read_csv(csv_path)[['name', 'seq']]
        elif csv_path.endswith('.tsv'):
            self.data = pd.read_table(csv_path, sep='\t')[['Entry','Sequence','EC number']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {'names': self.data.iloc[item][0], 'seqs': self.data.iloc[item][1],'ECnumber':self.data.iloc[item][2]}
