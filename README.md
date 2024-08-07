# ProteinMiningEvaluator
A pipeline for evaluating protein representation models on the task of protein mining.

#### 使用方法
```cmd

python run_benchmark.py --input 'xxxxxxx'
# your pkl file path
# pkl file must be same as: {'entry':entrys,'seqs':sequence,'embeddings':embeddings,'ecnumbers':ecnumbers'}
#                                                            embedding in embeddings must be tensor, not list

python run_analyze.py --level 4
# level ∈ [1,2,3,4]

```