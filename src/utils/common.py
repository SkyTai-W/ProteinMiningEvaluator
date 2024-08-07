import inspect
from Bio import SeqIO
from typing import List, Any
import pandas as pd
import torch
from Bio.Blast import NCBIXML
import yaml
import os


def world_size() -> int:
    return int(os.environ.get("SLURM_NTASKS", 1))


def global_rank() -> int:
    return int(os.environ.get("SLURM_PROCID", 0))


def local_rank() -> int:
    return int(os.environ.get("SLURM_LOCALID", 0))


def read_csv(*args, **kwargs):
    sep = kwargs.get("sep")
    if sep == "\t":
        return pd.read_table(*args, **kwargs)
    else:
        return pd.read_csv(*args, **kwargs)


def read_yaml(file):
    with open(file) as f:
        res = yaml.load(f.read(), Loader=yaml.FullLoader)
    return res


def read_seq_from_fasta(fasta_file_path):
    fasta = SeqIO.read(fasta_file_path, 'fasta')
    sequence = str(fasta.seq)
    return sequence


def write_seq_fo_fasta(sequence, path):
    SeqIO.write(sequence, open(path, "w+"), 'fasta')


def full_sequence(origin_sequence, raw_mutant):
    list_mutants = raw_mutant.split(";")
    sequence = origin_sequence
    for raw_mut in list_mutants:
        to = raw_mut[-1]
        pos = int(raw_mut[1:-1]) - 1
        assert sequence[pos] == raw_mut[
            0], "the original sequence is different to that in the mutant file in resid %d" % (pos + 1)
        sequence = sequence[:pos] + to + sequence[pos + 1:]
    return sequence


def read_mutant_seqs(origin_sequence: str, tsv_path) -> List[str]:
    """
    从一个序列和一个tsv文件中生成突变后的序列
    :param origin_sequence: 原始的序列
    :param tsv_path: tsv
    :return: 突变后的序列列表
    """
    if isinstance(tsv_path, list):
        sequences = []
        for each in tsv_path:
            sequences.extend(read_mutant_seqs(origin_sequence, each))
        return sequences
    table = pd.read_table(tsv_path)
    sequences = []
    for raw_mutant in table['mutant']:
        mutant_sequence = full_sequence(origin_sequence, raw_mutant)
        sequences.append(mutant_sequence)
    return sequences


def pad_sequence(sequences, batch_first=False, padding_value=0, padding_type='post'):
    r"""Pad a list of variable length Tensors with zero
    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.
    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.
    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])
    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *`` where `T` is the
            length of the longest sequence.
        Function assumes trailing dimensions and type of all the Tensors
            in sequences are same.
    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.
        padding_type: (string, 'pre' or 'post'): pad either before or after each sequence. Default: 'post'.
    Returns:
        Tensor of size ``T x B x *`` if batch_first is False
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    if padding_type == 'post':
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            if batch_first:
                out_tensor[i, :length, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor
    elif padding_type == 'pre':
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            if batch_first:
                out_tensor[i, -length:, ...] = tensor
            else:
                out_tensor[-length:, i, ...] = tensor
    else:
        raise ValueError("Padding_type must be 'post' or 'pre'")

    return out_tensor


def read_blast(file):
    result_handle = open(file, 'r')
    blast_record = NCBIXML.read(result_handle)
    subject_seqs = []
    for i in blast_record.alignments:
        if len(i.hsps[0].sbjct) != 0:
            subject_seqs.append(i.hsps[0].sbjct)
    return subject_seqs


def read_multilines_fasta(file):
    records = SeqIO.parse(file, format="fasta")
    seqs = []
    for each in records:
        seqs.append(each.seq)
    return seqs


def get_variable_name(var):
    """
    返回一个变量的变量名字符串。
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]
