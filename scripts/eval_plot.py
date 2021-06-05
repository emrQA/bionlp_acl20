from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import utils as utils

def evaluate_bert_emrqa(model, data, batch_size, tokenizer):
    actual_st = []
    actual_end = []
    
    pred_st = []
    pred_end = []
    for ind in tqdm(range(0, len(data), batch_size)):
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
    # ensemble_num flag: Was testing out an idea here. Can be ignored and kept as 1.
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

def plot_graphs(train_losses, fig_name, save_flag=False):
    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    # Train
    ax1.plot(train_losses, color='red',label='Train')

    fig.set_size_inches(11.5, 6.5)
    
    plt.xlabel("Epochs")
    plt.ylabel('Losses')
    plt.title("Loss Graph")
    plt.legend()
    plt.show()
    if(save_flag):
        plt.savefig('../graphs/'+fig_name)


def plot_graphs_acc(val_vals, start_or_end='start' ,save_flag=False):
    if(start_or_end=='start'):
        val_vals = [x[0] for x in val_vals]
    else:
        val_vals = [x[1] for x in val_vals]

    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    # Val
    ax1.plot(val_vals, color='blue',label='Val')
    fig.set_size_inches(11.5, 6.5)
    
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy Score')
    plt.title("Training Model")
    plt.legend()
    plt.show()