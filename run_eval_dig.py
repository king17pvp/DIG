from dig_bert import *
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
dataset = load_dataset('glue', 'sst2')['test']
data = list(zip(dataset['sentence'], dataset['label'], dataset['idx']))
print('Starting attribution computation...')
inputs = []
log_odds, comps, suffs, count, total_time = 0, 0, 0, 0, 0
print_step = 1
for row in tqdm(data):
    text = row[0]
    res_dig = dig_bert(text, model_name='distilbert-base-uncased-finetuned-sst-2-english', dataset='sst2', steps=30, factor=1, strategy='greedy', topk=20)
    log_odd, comp, suff = res_dig['log_odd'], res_dig['comp'], res_dig['suff']
    total_time += res_dig['time']
    log_odds += res_dig['log_odd']
    comps += res_dig['comp']
    suffs += res_dig['suff']
    count += 1

    # print the metrics
    if count % print_step == 0:
        print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 'Sufficiency: ', np.round(suffs / count, 4))

print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 'Sufficiency: ', np.round(suffs / count, 4))