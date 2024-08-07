import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
from src.utils.common import read_seq_from_fasta


def main(args):
    root_path = 'benchmark'
    subdirectories = os.listdir(root_path)

    swiss_path = 'data/SwissProt.tsv'
    df = pd.read_table(swiss_path, sep='\t')

    methods = [name for name in os.listdir(root_path + '/' + subdirectories[0]) if
               os.path.isdir(os.path.join(root_path + '/' + subdirectories[0], name))]

    blast_result = []

    for entry in subdirectories:

        result_path = root_path + '/' + entry + '/BLAST' + '/result.tsv'

        query_ec_number = eval(str(df.loc[df['Entry'] == entry]['EC number'].values))

        results = pd.read_table(result_path, sep='\t')['Accession']

        for result in results:
            blast_ec_number = eval(str(df.loc[df['Entry'] == result]['EC number'].values))

            query_tmp = ['.'.join(i.split('.')[:args.level]) for i in query_ec_number]
            blast_tmp = ['.'.join(i.split('.')[:args.level]) for i in blast_ec_number]

            query_ec_number_set = set(query_tmp)
            blast_ec_number_set = set(blast_tmp)
            intersection = query_ec_number_set & blast_ec_number_set
            if len(intersection) <= 0:
                continue

            blast_result.append(result)

    hit_rate = []

    for method in methods:

        hit_rate_topk = []

        method_result = []

        for query_entry in subdirectories:

            count = 0

            json_path = root_path + '/' + query_entry + '/' + query_entry + '.json'

            with open(json_path, 'r') as file:
                data = json.load(file)

            query_ec_number = eval(str(data['EC number']))

            query_tmp = ['.'.join(i.split('.')[:args.level]) for i in query_ec_number]

            query_ec_number_set = set(query_tmp)

            result_path = root_path + '/' + query_entry + '/' + method + '/result.tsv'

            results = pd.read_table(result_path, sep='\t')['ECnumber']

            for idx, result in enumerate(results):
                result_ec_number = eval(str(result))

                result_tmp = ['.'.join(i.split('.')[:args.level]) for i in result_ec_number]

                result_ec_number_set = set(result_tmp)

                intersection = query_ec_number_set & result_ec_number_set

                if len(intersection) > 0:
                    count = count + 1
                    method_result.append(result)

                if (idx + 1) % 50 == 0:
                    hit_rate_topk.append(count / (idx + 1))

        hit_rate.append([method] + hit_rate_topk)

        tmp1 = set(blast_result)
        tmp2 = set(method_result)

        plt.figure()  # 创建新图形
        sns.set_style('darkgrid')
        venn2([tmp1, tmp2], ('BLAST', method))  # 绘制 Venn 图
        plt.title('BLAST' + '  &&  ' + method + '  (Level : ' + str(args.level) + ')')  # 设置标题
        sns.despine(left=True, bottom=True)  # 去掉多余的轴线

        pic_path = 'BLAST' + '_' + method + '_venn_level' + str(args.level) + '_top' + str(
            args.topk) + '.png'
        plt.savefig(pic_path)

        # plt.show()  # 显示图形

    output = pd.DataFrame(hit_rate,
                          columns=['method', 'top50_level' + str(args.level), 'top100_level' + str(args.level),
                                   'top150_level' + str(args.level), 'top200_level' + str(args.level),
                                   'top250_level' + str(args.level)])
    output.to_csv('hit_rate_level' + str(args.level) + '.tsv', sep='\t')

    sns.set_palette("Dark2")

    plt.figure(figsize=(14, 9))

    plt.rcParams['font.weight'] = 'bold'

    for method in methods:
        sns.lineplot(data=output[output['method'] == method].values[0][1:], label=method, linewidth=1)

    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=[30, 50, 100, 150, 200, 250], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Hit Rate', fontsize=18)
    plt.xlabel('TopK', fontsize=18)
    plt.title('Hit rate of modules & methods ' + '(Match Level : level' + str(args.level) + ')', fontsize=18)

    pic_path = 'Hit_rate_of_modules_and_methods_' + 'level' + str(args.level) + '.png'

    plt.savefig(pic_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input topk')

    parser.add_argument('-l', '--level', type=int, default=4, help='Input match level(a number)')

    args = parser.parse_args()

    main(args)
