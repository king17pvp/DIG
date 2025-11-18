from dig_bert import *
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='distilbert', help='Model name or path')
args = parser.parse_args()
model = args.model
if model == 'distilbert':
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
elif model == 'bert':
    model_name = "textattack/bert-base-uncased-SST-2"
elif model == 'roberta':
    model_name = "textattack/roberta-base-SST-2"
dataset = load_dataset('glue', 'sst2')['test']
data = list(zip(dataset['sentence'], dataset['label'], dataset['idx']))
print('Starting attribution computation...')
inputs = []
log_odds, comps, suffs, count, total_time = 0, 0, 0, 0, 0
print_step = 100
for row in tqdm(data):
    text = row[0]
    res_dig = dig_bert(text, model_name=model_name, dataset='sst2', steps=30, factor=0, strategy='greedy', topk=20)
    log_odd, comp, suff = res_dig['log_odd'], res_dig['comp'], res_dig['suff']
    total_time += res_dig['time']
    log_odds += res_dig['log_odd']
    comps += res_dig['comp']
    suffs += res_dig['suff']
    count += 1

    # print the metrics
    if count % print_step == 0:
        print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 'Sufficiency: ', np.round(suffs / count, 4), 'Time: ', np.round(total_time / count, 4))

print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 'Sufficiency: ', np.round(suffs / count, 4), 'Time: ', np.round(total_time / count, 4))