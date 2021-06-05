import pandas as pd
import numpy as np
import pickle
from random import shuffle
from tqdm import tqdm_notebook as tqdm
import math
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import utils as utils

import re
import string
import collections

def evaluate_bert_emrqa(model, data, batch_size, tokenizer):
    actual_st = []
    actual_end = []
    
    pred_st = []
    pred_end = []
    for ind in tqdm(range(0, len(data), batch_size)):
        try:
            current_batch = utils.batchify_bert_emrqa(data[ind:ind+batch_size],
                                                      tokenizer)

            all_input_ids = torch.tensor(current_batch['sents'], dtype=torch.long)
            all_input_mask = torch.tensor(current_batch['masking_ids'], dtype=torch.long)
            all_segment_ids = torch.tensor(current_batch['segment_ids'], dtype=torch.long)
            all_start_positions = torch.tensor(current_batch['start_positions'], dtype=torch.long)
            all_end_positions = torch.tensor(current_batch['end_positions'], dtype=torch.long)

            logits = model(all_input_ids, all_segment_ids, all_input_mask) 
            _, st_inds = torch.max(logits[0], dim=1)
            _, end_inds = torch.max(logits[1], dim=1)
            
            actual_st+=all_start_positions.data.tolist()
            pred_st+=st_inds.data.tolist()
            
            actual_end+=all_end_positions.data.tolist()
            pred_end+=end_inds.data.tolist()
        except:
            print("The set of batch that didnt't work out", (ind, ind+batch_size))
    start_accuracy = accuracy_score(actual_st, pred_st)
    end_accuracy = accuracy_score(actual_end, pred_end)
    actual_and_predicted_values = (actual_st, actual_end, pred_st, pred_end)
    return start_accuracy, end_accuracy, actual_and_predicted_values

def evaluate_bert_emrqa_ernie(model, data, batch_size, tokenizer, entity2id, logical2ix):
    actual_st = []
    actual_end = []
    
    pred_st = []
    pred_end = []
    for ind in tqdm(range(0, len(data), batch_size)):
        current_batch = utils.batchify_bert_emrqa_ernie(data[ind:ind+batch_size],
                                        tokenizer, 
                                        entity2id,
                                        logical2ix)

        all_input_ids = torch.tensor(current_batch['sents'], dtype=torch.long)
        all_input_mask = torch.tensor(current_batch['masking_ids'], dtype=torch.long)

        all_ent_ids = torch.tensor(current_batch['entities'], dtype=torch.long)
        all_ent_masks = torch.tensor(current_batch['entity_masking_ids'], dtype=torch.long)

        all_segment_ids = torch.tensor(current_batch['segment_ids'], dtype=torch.long)
        all_start_positions = torch.tensor(current_batch['start_positions'], dtype=torch.long)
        all_end_positions = torch.tensor(current_batch['end_positions'], dtype=torch.long)
        with torch.no_grad():
            logits, lf_logits = model(input_ids=all_input_ids, 
                         token_type_ids=all_segment_ids, 
                         attention_mask=all_input_mask, 
                         input_ents_idx=all_ent_ids, 
                         ent_mask=all_ent_masks)
        _, st_inds = torch.max(logits[0], dim=1)
        _, end_inds = torch.max(logits[1], dim=1)
        
        actual_st+=all_start_positions.data.tolist()
        pred_st+=st_inds.data.tolist()
        
        actual_end+=all_end_positions.data.tolist()
        pred_end+=end_inds.data.tolist()
    start_accuracy = accuracy_score(actual_st, pred_st)
    end_accuracy = accuracy_score(actual_end, pred_end)
    actual_and_predicted_values = (actual_st, actual_end, pred_st, pred_end)
    return start_accuracy, end_accuracy, actual_and_predicted_values

def evaluate_boolQ(model, data, batch_size, tokenizer, label2ix, entity2id=None):
    actual_labels = []
    pred_labels = []
    
    for ind in tqdm(range(0, len(data), batch_size)):
        current_batch = utils.batchify_boolQ(data[ind:ind+batch_size],
                                     tokenizer=tokenizer,
                                     label2ix=label2ix)

        all_input_ids = torch.tensor(current_batch['sents'], dtype=torch.long)
        all_input_mask = torch.tensor(current_batch['masking_ids'], dtype=torch.long)
        all_segment_ids = torch.tensor(current_batch['segment_ids'], dtype=torch.long)
        all_labels = torch.tensor(current_batch['labels'], dtype=torch.long)

        logits = model(input_ids=all_input_ids, 
                      token_type_ids=all_segment_ids, 
                      attention_mask=all_input_mask)

        _, preds = torch.max(logits, dim=1)
        
        actual_labels+=all_labels.data.tolist()
        pred_labels+=preds.data.tolist()

    accuracy = accuracy_score(actual_labels, pred_labels)
    prf_vals = utils.get_metrics(actual_outputs=actual_labels, preds=pred_labels)
    actual_and_predicted_values = (actual_labels, pred_labels)
    return accuracy, prf_vals, actual_and_predicted_values

def evaluate_boolQ_ernie(model, data, batch_size, tokenizer, label2ix, entity2id=None, title_label2id=None):
    actual_labels = []
    pred_labels = []
    
    for ind in tqdm(range(0, len(data), batch_size)):
        current_batch = utils.batchify_boolQ_ernie(data[ind:ind+batch_size], 
                                                   tokenizer=tokenizer, 
                                                   label2ix=label2ix, 
                                                   entity2id=entity2id, 
                                                   title2ix = title_label2id)

        all_input_ids = torch.tensor(current_batch['sents'], dtype=torch.long)
        all_input_mask = torch.tensor(current_batch['masking_ids'], dtype=torch.long)

        all_ent_ids = torch.tensor(current_batch['entities'], dtype=torch.long)
        all_ent_masks = torch.tensor(current_batch['entity_masking_ids'], dtype=torch.long)

        all_segment_ids = torch.tensor(current_batch['segment_ids'], dtype=torch.long)
        all_title_label = torch.tensor(current_batch['titles'], dtype=torch.long)
        all_labels = torch.tensor(current_batch['labels'], dtype=torch.long)

        logits, _ = model(input_ids=all_input_ids, 
                                     token_type_ids=all_segment_ids, 
                                     attention_mask=all_input_mask, 
                                     input_ent_idx=all_ent_ids, 
                                     ent_mask=all_ent_masks)

        _, preds = torch.max(logits, dim=1)
        
        actual_labels+=all_labels.data.tolist()
        pred_labels+=preds.data.tolist()

    accuracy = accuracy_score(actual_labels, pred_labels)
    prf_vals = utils.get_metrics(actual_outputs=actual_labels, preds=pred_labels)
    actual_and_predicted_values = (actual_labels, pred_labels)
    return accuracy, prf_vals, actual_and_predicted_values

def evaluate_boolQ_proj(model, data, batch_size, tokenizer, label2ix, entity2id=None, title_label2id=None, device=None):
    actual_labels = []
    pred_labels = []
    
    for ind in tqdm(range(0, len(data), batch_size)):
        current_batch = utils.batchify_boolQ_ernie(data[ind:ind+batch_size], 
                                                   tokenizer=tokenizer, 
                                                   label2ix=label2ix, 
                                                   entity2id=entity2id, 
                                                   title2ix = title_label2id)

        all_input_ids = torch.tensor(current_batch['sents'], dtype=torch.long)
        all_input_mask = torch.tensor(current_batch['masking_ids'], dtype=torch.long)

        all_ent_ids = torch.tensor(current_batch['entities'], dtype=torch.long)
        all_ent_masks = torch.tensor(current_batch['entity_masking_ids'], dtype=torch.long)

        all_segment_ids = torch.tensor(current_batch['segment_ids'], dtype=torch.long)
        all_title_label = torch.tensor(current_batch['titles'], dtype=torch.long)
        all_labels = torch.tensor(current_batch['labels'], dtype=torch.long)
        all_lens = torch.tensor(current_batch['lens'], dtype=torch.long)

        logits = model(all_input_ids, 
             token_type_ids=all_segment_ids, 
             attention_mask=all_input_mask, 
             input_ent_idx=all_ent_ids, 
             ent_mask=all_ent_masks,
             logical_form_ids=all_title_label,
             sent_lens = all_lens,
             device = device)

        _, preds = torch.max(logits, dim=1)
        
        actual_labels+=all_labels.data.tolist()
        pred_labels+=preds.data.tolist()

    accuracy = accuracy_score(actual_labels, pred_labels)
    prf_vals = utils.get_metrics(actual_outputs=actual_labels, preds=pred_labels)
    actual_and_predicted_values = (actual_labels, pred_labels)
    return accuracy, prf_vals, actual_and_predicted_values

def evaluate_boolQ_ernie_get_lf(model, data, batch_size, tokenizer, label2ix, entity2id=None, title_label2id=None):
    actual_labels = []
    pred_labels = []
    actual_titles = []
    pred_titles = []
    for ind in tqdm(range(0, len(data), batch_size)):
        current_batch = utils.batchify_boolQ_ernie(data[ind:ind+batch_size], 
                                                   tokenizer=tokenizer, 
                                                   label2ix=label2ix, 
                                                   entity2id=entity2id, 
                                                   title2ix = title_label2id)

        all_input_ids = torch.tensor(current_batch['sents'], dtype=torch.long)
        all_input_mask = torch.tensor(current_batch['masking_ids'], dtype=torch.long)

        all_ent_ids = torch.tensor(current_batch['entities'], dtype=torch.long)
        all_ent_masks = torch.tensor(current_batch['entity_masking_ids'], dtype=torch.long)

        all_segment_ids = torch.tensor(current_batch['segment_ids'], dtype=torch.long)
        all_title_label = torch.tensor(current_batch['titles'], dtype=torch.long)
        
        all_labels = torch.tensor(current_batch['labels'], dtype=torch.long)

        logits, logits_title = model(input_ids=all_input_ids, 
                                     token_type_ids=all_segment_ids, 
                                     attention_mask=all_input_mask, 
                                     input_ent_idx=all_ent_ids, 
                                     ent_mask=all_ent_masks)

        _, preds = torch.max(logits, dim=1)
        _, preds_title = torch.max(logits_title, dim=1)
        
        
        actual_labels+=all_labels.data.tolist()
        pred_labels+=preds.data.tolist()
        actual_titles+=all_title_label.data.tolist()
        pred_titles+=preds_title.data.tolist()

    accuracy = accuracy_score(actual_labels, pred_labels)
    prf_vals = utils.get_metrics(actual_outputs=actual_labels, preds=pred_labels)
    actual_and_predicted_values = (actual_labels, pred_labels)
    
    prf_titles = utils.get_metrics(actual_outputs=actual_titles, preds=pred_titles)
    return accuracy, prf_vals, actual_and_predicted_values, (prf_titles, (actual_titles, pred_titles))


def evaluate_bert_emrqa_ernie_multitask(model, data, batch_size, tokenizer, entity2id, logical2ix, device, ensemble_num = 1):
    actual_st = []
    actual_end = []
    
    pred_st = []
    pred_end = []
    for ind in tqdm(range(0, len(data), batch_size)):
        current_batch = utils.batchify_bert_emrqa_ernie(data[ind:ind+batch_size],
                                                        tokenizer, 
                                                        entity2id,
                                                        logical2ix)

        all_input_ids = torch.tensor(current_batch['sents'], dtype=torch.long)
        all_input_mask = torch.tensor(current_batch['masking_ids'], dtype=torch.long)

        all_ent_ids = torch.tensor(current_batch['entities'], dtype=torch.long)
        all_ent_masks = torch.tensor(current_batch['entity_masking_ids'], dtype=torch.long)

        all_segment_ids = torch.tensor(current_batch['segment_ids'], dtype=torch.long)
        all_start_positions = torch.tensor(current_batch['start_positions'], dtype=torch.long)
        all_end_positions = torch.tensor(current_batch['end_positions'], dtype=torch.long)

        all_logical_forms = torch.tensor(current_batch['logical_forms'], dtype=torch.long).to(device)
        
        for i in range(ensemble_num):
            tmp_logits, tmp_lf_logits = model(input_ids=all_input_ids, 
                                         token_type_ids=all_segment_ids, 
                                         attention_mask=all_input_mask, 
                                         input_ents_idx=all_ent_ids, 
                                         ent_mask=all_ent_masks)
            if(i==0):
                logits = [tmp_logits[0], tmp_logits[1]]
            else:
                logits[0] = logits[0]+tmp_logits[0]
                logits[1] = logits[1]+tmp_logits[1]
        logits[0] = logits[0]/ensemble_num
        logits[1] = logits[1]/ensemble_num
        
        _, st_inds = torch.max(logits[0], dim=1)
        _, end_inds = torch.max(logits[1], dim=1)
        
        actual_st+=all_start_positions.data.tolist()
        pred_st+=st_inds.data.tolist()
        
        actual_end+=all_end_positions.data.tolist()
        pred_end+=end_inds.data.tolist()
    start_accuracy = accuracy_score(actual_st, pred_st)
    end_accuracy = accuracy_score(actual_end, pred_end)
    actual_and_predicted_values = (actual_st, actual_end, pred_st, pred_end)
    return start_accuracy, end_accuracy, actual_and_predicted_values

######################## PLOT GRAPHS ############################
def exact_match(actual_st, actual_end, pred_st, pred_end):
    match = 0
    for ind in range(len(actual_st)):
        if((actual_st[ind]==pred_st[ind]) and (actual_end[ind]==pred_end[ind])):
            match+=1
    return (match*100.0)/len(actual_st)

def get_f1(actual_st, actual_end, pred_st, pred_end):
    all_f1 = []
    for ind in range(len(actual_st)):
        tmp = compute_f1(actual_st[ind], actual_end[ind], pred_st[ind], pred_end[ind])
        all_f1.append(tmp)
    return sum(all_f1)/len(all_f1)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_f1(actual_st, actual_end, pred_st, pred_end):
    gold_toks = [x for x in range(actual_st, actual_end+1)]
    pred_toks = [x for x in range(pred_st, pred_end+1)]
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def plot_loss(train_losses, fig_name, save_flag=False):
    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    # Train
    ax1.plot(train_losses, color='red',label='Train')
    fig.set_size_inches(11.5, 6.5)
    plt.xlabel("Epochs")
    plt.ylabel('Losses')
    plt.title("Model")
    plt.legend()
    plt.show()
    if(save_flag):
        plt.savefig('../graphs/'+fig_name)

def plot_graphs(train_losses, val_losses, test_losses, fig_name, save_flag=False):
    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    # Train
    ax1.plot(train_losses, color='red',label='Train')
    # Val
    ax1.plot(val_losses, color='blue',label='Val')
    # Test
    ax1.plot(test_losses, color='green',label='Test')

    fig.set_size_inches(11.5, 6.5)
    
    plt.xlabel("Epochs")
    plt.ylabel('Losses')
    plt.title("Model")
    plt.legend()
    plt.show()
    if(save_flag):
        plt.savefig('../graphs/'+fig_name)


def plot_graphs_acc(train_vals, val_vals, test_vals, start_or_end='start' ,save_flag=False):
    if(start_or_end=='start'):
        train_vals = [x[0] for x in train_vals]
        val_vals = [x[0] for x in val_vals]
        test_vals = [x[0] for x in test_vals]
    else:
        train_vals = [x[1] for x in train_vals]
        val_vals = [x[1] for x in val_vals]
        test_vals = [x[1] for x in test_vals]

    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    # Train
    ax1.plot(train_vals, color='red',label='Train')
    # Val
    ax1.plot(val_vals, color='blue',label='Val')
    # Test
    ax1.plot(test_vals, color='green',label='Test')
    fig.set_size_inches(11.5, 6.5)
    
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy Score')
    plt.title("Whatever model you were learning!")
    plt.legend()
    plt.show()


def plot_graph_fscore(dict_prf):
    total_epochs = len(dict_prf['train'])
    train_vals = [dict_prf['train']['epoch_'+str(x)][0]['ma-f'][0] for x in range(total_epochs)]
    val_vals = [dict_prf['dev']['epoch_'+str(x)][0]['ma-f'][0] for x in range(total_epochs)]
    test_vals = [dict_prf['test']['epoch_'+str(x)][0]['ma-f'][0] for x in range(total_epochs)]
    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    # Train
    ax1.plot(train_vals, color='red',label='Train')
    # Val
    ax1.plot(val_vals, color='blue',label='Val')
    # Test
    ax1.plot(test_vals, color='green',label='Test')
    fig.set_size_inches(11.5, 6.5)
    
    plt.xlabel("Epochs")
    plt.ylabel('F-score')
    plt.title("F-score Performance")
    plt.legend()
    plt.show()
    ind_highest_val = np.argmax(val_vals)
    print("-x-"*5 + str(ind_highest_val) + "-x-"*5)
    print("Train")
    print(dict_prf['train']['epoch_'+str(ind_highest_val)])
    print("=="*4)
    print("Val")
    print(dict_prf['dev']['epoch_'+str(ind_highest_val)])
    print("=="*4)
    print("Test")
    print(dict_prf['test']['epoch_'+str(ind_highest_val)])
    print("=="*4)