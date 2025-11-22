from dig_bert import *
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse
import torch
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='distilbert', help='Model name or path')
parser.add_argument('--dataset', choices=['sst2', 'imdb', 'rotten'])
parser.add_argument('--strategy', default = 'greedy', choices=['greedy', 'maxcount'], help='The algorithm to find the next anchor point')
args = parser.parse_args()
model = args.model
if model == 'distilbert':
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
elif model == 'bert':
    model_name = "textattack/bert-base-uncased-SST-2"
elif model == 'roberta':
    model_name = "textattack/roberta-base-SST-2"
    
if args.dataset == 'imdb':
    dataset	= load_dataset('imdb')['test']
    data	= list(zip(dataset['text'], dataset['label']))
    data	= random.sample(data, 2000)
elif args.dataset == 'sst2':
    dataset	= load_dataset('glue', 'sst2')['test']
    data	= list(zip(dataset['sentence'], dataset['label'], dataset['idx']))
elif args.dataset == 'rotten':
    dataset	= load_dataset('rotten_tomatoes')['test']
    data	= list(zip(dataset['text'], dataset['label']))
print('Starting attribution computation...')
inputs = []
log_odds, comps, suffs, count, total_time = 0, 0, 0, 0, 0
print_step = 100
for row in tqdm(data):
    text = row[0]
    res_dig = dig_bert(text, model_name=model_name, dataset=args.dataset, steps=30, factor=0, strategy=args.strategy, topk=20)
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