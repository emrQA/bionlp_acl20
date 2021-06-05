import pandas as pd
import numpy as np
import pickle
from random import shuffle
from tqdm import tqdm_notebook as tqdm
import math
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
from sklearn.metrics import precision_recall_fscore_support
# 1 starts the process on GPU-0
# 0 starts the process on GPU-2
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
from pytorch_pretrained_bert.tokenization import whitespace_tokenize
# from gensim.models import KeyedVectors

# word2vec = KeyedVectors.load_word2vec_format('/home/lif/fasttext_ade_meddra_snomed_lower_0norm_200d.vec')

######################## PREPARE DICTS #######################################
def prepare_dicts_classification(data_, questions, tokenizer):
    word_to_ix = {}
    tag_to_ix = {}
    class_to_ix = {}
    relevance_to_ix = {}
    for each_ in tqdm(data_):
        curr_sent = tokenizer(each_[3])
        each_[3] = curr_sent
        for each_token in curr_sent: 
            if(each_token not in word_to_ix):
                word_to_ix[each_token] = len(word_to_ix)
    for each_key in questions.keys():
        for each_token in questions[each_key]:
            if(each_token not in word_to_ix):
                word_to_ix[each_token] = len(word_to_ix)
    word_to_ix['<START>'] = len(word_to_ix)
    word_to_ix['<END>'] = len(word_to_ix)
    word_to_ix['<UNK>'] = len(word_to_ix)
    word_to_ix['<PAD>'] = len(word_to_ix)
    word_to_ix['<SEP>'] = len(word_to_ix)
    
    relevance_to_ix['Non-Relevant'] = 0
    relevance_to_ix['Relevant'] = 1
    
    class_to_ix['Yes'] = 0
    class_to_ix['No'] = 1
    class_to_ix['DK'] = 2
    # class_to_ix['False'] = 3
    
    
    to_ix = {}
    to_ix['word2ix'] = word_to_ix
    to_ix['class2ix'] = class_to_ix
    to_ix['relevance2ix'] = relevance_to_ix
    return to_ix

def prepare_dicts_oracle(data_, questions, tokenizer):
    word_to_ix = {}
    tag_to_ix = {}
    class_to_ix = {}
    relevance_to_ix = {}
    drug_to_ix = {}

    for each_k in questions.keys():
        if(each_k not in drug_to_ix):
            drug_to_ix[each_k] = len(drug_to_ix)
    for each_ in data_:
        for ind, each_sent in enumerate(each_[0]):    
            curr_sent = tokenizer(each_sent[3])
            each_sent[3] = curr_sent
            for each_token in curr_sent: 
                if(each_token not in word_to_ix):
                    word_to_ix[each_token] = len(word_to_ix)
    for each_key in questions.keys():
        for each_token in questions[each_key]:
            if(each_token not in word_to_ix):
                word_to_ix[each_token] = len(word_to_ix)
    word_to_ix['<START>'] = len(word_to_ix)
    word_to_ix['<END>'] = len(word_to_ix)
    word_to_ix['<UNK>'] = len(word_to_ix)
    word_to_ix['<PAD>'] = len(word_to_ix)
    word_to_ix['<SEP>'] = len(word_to_ix)
    
    relevance_to_ix['Non-Relevant'] = 0
    relevance_to_ix['Relevant'] = 1
    
    class_to_ix['Yes'] = 0
    class_to_ix['No'] = 1
    class_to_ix['DK'] = 2
    # class_to_ix['False'] = 3
    
    to_ix = {}
    to_ix['word2ix'] = word_to_ix
    to_ix['class2ix'] = class_to_ix
    to_ix['relevance2ix'] = relevance_to_ix
    to_ix['drug2ix'] = drug_to_ix
    return to_ix

def prepare_dicts_one_hot_encoder(data_, questions, tokenizer):
    word_to_ix = {}
    for each_ in data_:
        for ind, each_sent in enumerate(each_[0]):    
            curr_sent = tokenizer(each_sent[3])
            for each_token in curr_sent: 
                if(each_token not in word_to_ix):
                    word_to_ix[each_token] = len(word_to_ix)
    for each_key in questions.keys():
        for each_token in questions[each_key]:
            if(each_token not in word_to_ix):
                word_to_ix[each_token] = len(word_to_ix)
    word_to_ix['<UNK>'] = len(word_to_ix)
    return word_to_ix

def get_word2vec_numpy_array(word_to_ix, embed_dim):
    vocab = word2vec.vocab
    embeddings = [np.random.rand(embed_dim) for x in range(len(word_to_ix))]
    cnt_ = 0
    for each_word in word_to_ix.keys():
        if(each_word in vocab):
            embeddings[word_to_ix[each_word]] = word2vec[each_word]
            cnt_+=1
    print("Total Words: "+str(len(word_to_ix))+" Found: "+str(cnt_))
    return np.asarray(embeddings)


def prepare_entities_to_ix(list_ents):
    dict_ = {}
    for each_ent in list_ents:
        dict_[each_ent] = len(dict_)
    return dict_

def prepare_logical_forms_to_ix(list_):
    all_logical_forms = []
    dict_ = {}
    for each_ in list_:
        if(each_['logical_form_template'] not in dict_):
            dict_[each_['logical_form_template']] = len(dict_)
    return dict_

################### GET DISTR ##########################
def get_class_distr(data_, ind=-3):
    no_cnt = 0 
    yes_cnt = 0
    dk_cnt = 0
    false_cnt = 0
    for each_data in data_:
        if('Yes' == each_data[ind]):
            yes_cnt+=1
        elif('No' == each_data[ind]):
            no_cnt+=1
        elif('False' == each_data[ind]):
            false_cnt+=1
        elif('DK' == each_data[ind]):
            dk_cnt+=1
    return {"Yes":yes_cnt,"No": no_cnt,"False":false_cnt, "DK":dk_cnt}

def get_relevance_distr(data_, ind=-2):
    relevant_cnt = 0
    not_relevant_cnt = 0
    for each_data in data_:
        if('Relevant'==each_data[ind]):
            relevant_cnt+=1
        elif('Non-Relevant'==each_data[ind]):
            not_relevant_cnt+=1
    return {'Relevant':relevant_cnt, 'Non-Relevant':not_relevant_cnt}

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


################### GET DATA ################################
    
def downsample_data(data, per_down, score_ind = 5):
    rem_per = 1-per_down
    data_false = [x for x in data if(x[score_ind]=='Non-Relevant')]
    data_yes = [x for x in data if(x[score_ind]!='Non-Relevant')]
    print("Len Data:",len(data_false))
    rem_size = int(len(data_false)*rem_per)
    rem_false_data = random.sample(data_false, rem_size)
    final_list = rem_false_data+data_yes
    final_list = random.sample(final_list, len(final_list))
    return final_list

######################## SEQUENCE AND BATCHIFY ##############################
def get_sequence(sentences, to_ix, LIMIT_LEN = 150):
    max_len = 0
    batch_sents = []
    for each_sentence in sentences:
        batch_sent = []
        for x in each_sentence:
            if(x not in to_ix['word2ix']):
                batch_sent.append(to_ix['word2ix']['<UNK>'])
            else:
                batch_sent.append(to_ix['word2ix'][x])
        if(len(batch_sent)>max_len):
            max_len = len(batch_sent)
        batch_sents.append(batch_sent)
    if(max_len>LIMIT_LEN):
        max_len = LIMIT_LEN
    all_lens = [len(x[:max_len]) for x in batch_sents]
    all_lens = [x+2 for x in all_lens]
    batch_sents = [[to_ix['word2ix']['<START>']]+batch_sents[ind][:max_len]+[to_ix['word2ix']['<END>']]+[to_ix['word2ix']['<PAD>'] for x in range(max_len-len(batch_sents[ind]))] for ind in range(len(batch_sents))]

    descend_indices = list(np.argsort(all_lens))
    descend_indices.reverse()
    sorted_sents = []
    sorted_lens = []
    for ind in descend_indices:
        sorted_sents.append(batch_sents[ind])
        sorted_lens.append(all_lens[ind])
    return sorted_sents, sorted_lens, descend_indices

def batchify_oracle(all_datapoints, to_ix, answer, sc_weights, curr_question):
    main_sents = []
    main_scores = []
    main_sc_weights = []
    for each_datapoint in all_datapoints:
        main_sents.append(curr_question + ['<SEP>'] + each_datapoint[3])
    sorted_sents, sorted_lens, descend_indices = get_sequence(main_sents, to_ix)
    sorted_scores = []
    sorted_sc_weights = []
    
    sorted_scores.append(to_ix['class2ix'][answer])
    sorted_sc_weights.append(sc_weights[answer])
    
    dict_ = {'sents':sorted_sents, 
            'lens':sorted_lens,
            'scores':sorted_scores,
            'sc_weights':sorted_sc_weights}
    return dict_

def batchify_oracle_proj(all_datapoints, to_ix, answer, sc_weights, curr_question, drug_ques_id):
    main_sents = []
    main_scores = []
    main_sc_weights = []
    main_drug_ques_ids = []
    for each_datapoint in all_datapoints:
        main_sents.append(curr_question + ['<SEP>'] + each_datapoint[3])
        main_drug_ques_ids.append(to_ix['drug2ix'][drug_ques_id])
    sorted_sents, sorted_lens, descend_indices = get_sequence(main_sents, to_ix)
    sorted_scores = []
    sorted_sc_weights = []
    
    sorted_scores.append(to_ix['class2ix'][answer])
    sorted_sc_weights.append(sc_weights[answer])
    
    dict_ = {'sents':sorted_sents, 
            'lens':sorted_lens,
            'scores':sorted_scores,
            'sc_weights':sorted_sc_weights,
            'drug_ques_ids':main_drug_ques_ids}
    return dict_

def batchify_bilstm_classification(all_datapoints, to_ix, answer, sc_weights, questions):
    main_sents = []
    main_scores = []
    main_sc_weights = []
    main_drug_ques_ids = []
    for each_datapoint in all_datapoints:
        curr_question = questions[each_datapoint[6]]
        main_sents.append(curr_question + ['<SEP>'] + each_datapoint[3])
        main_scores.append(to_ix['relevance2ix'][each_datapoint[5]])
        main_sc_weights.append(sc_weights[each_datapoint[5]])
        main_drug_ques_ids.append(each_datapoint[6])
    sorted_sents, sorted_lens, descend_indices = get_sequence(main_sents, to_ix)
    sorted_scores = []
    sorted_sc_weights = []
    sorted_drug_ques_ids = []

    for ind in descend_indices:
        sorted_scores.append(main_scores[ind])
        sorted_sc_weights.append(main_sc_weights[ind])
        sorted_drug_ques_ids.append(main_drug_ques_ids[ind])
    
    dict_ = {'sents':sorted_sents, 
            'lens':sorted_lens,
            'scores':sorted_scores,
            'sc_weights':sorted_sc_weights,
            'drug_ques_ids':sorted_drug_ques_ids}
    return dict_

def batchify_oracle_bert(all_datapoints, answer, sc_weights, curr_question, tokenizer, class2ix, LIMIT_LEN = 250):
    ques = []
    sent = []

    ques_sent = []
    segment_ids = []
    masking_ids = []
    main_scores = []
    main_sc_weights = []

    cls_id = ['[CLS]']
    sep_id = ['[SEP]']
    pad_id = '[PAD]'
    max_len = 0 
    for ind, each_datapoint in enumerate(all_datapoints):
        ques.append(cls_id + tokenizer.tokenize(curr_question) + sep_id)
        sent.append(tokenizer.tokenize(each_datapoint[3])+ sep_id)
        curr_len = len(ques[ind])+len(sent[ind])
        if(curr_len>max_len):
            max_len = curr_len
        segment_ids.append([0 for x in range(len(ques[ind]))] + [1 for x in range(len(sent[ind]))])
        masking_ids.append([1 for x in range(curr_len)])
    if(max_len > LIMIT_LEN):
        max_len = LIMIT_LEN

    for ind in range(len(ques)):
        curr_len = len(ques[ind]) + len(sent[ind])
        tmp_q_sent = tokenizer.convert_tokens_to_ids(ques[ind])+tokenizer.convert_tokens_to_ids(sent[ind])
        ques_sent.append( tmp_q_sent[:max_len] + [0 for x in range(max_len - curr_len)] )
        segment_ids[ind] = segment_ids[ind][:max_len] + [1 for x in range(max_len-curr_len)]
        masking_ids[ind] = masking_ids[ind][:max_len] + [0 for x in range(max_len-curr_len)]

    sorted_scores = []
    sorted_sc_weights = []
    
    sorted_scores.append(class2ix[answer])
    sorted_sc_weights.append(sc_weights[answer])
    
    dict_ = {'sents':ques_sent, 
            'segment_ids':segment_ids,
            'masking_ids':masking_ids,
            'scores':sorted_scores,
            'sc_weights':sorted_sc_weights}
    return dict_

def get_one_hot_vector(tokenized_list, word_to_ix):
    vect = [0 for i in range(len(word_to_ix))]
    for each_token in tokenized_list:
        if(each_token in word_to_ix):
            vect[word_to_ix[each_token]] = 1
        else:
            vect[word_to_ix['<UNK>']] = 1
    return vect

def batchify_oracle_bilstm_bert(all_datapoints, answer, sc_weights, questions, curr_drug_ques_id, 
                                tokenizer, class2ix, drug2ix, tokenizer_one_hot_encoder, word_to_ix, LIMIT_LEN = 250):
    ques = []
    sent = []

    ques_sent = []
    segment_ids = []
    masking_ids = []
    main_scores = []
    main_sc_weights = []
    main_drug_ques_ids = []
    main_one_hot_vectors = []

    cls_id = ['[CLS]']
    sep_id = ['[SEP]']
    pad_id = '[PAD]'
    max_len = 0 

    curr_question = " ".join(questions[curr_drug_ques_id])
    for ind, each_datapoint in enumerate(all_datapoints):
        curr_ques_sent_tokenized = tokenizer_one_hot_encoder(curr_question)+tokenizer_one_hot_encoder(each_datapoint[3])
        curr_one_hot_vector = get_one_hot_vector(curr_ques_sent_tokenized, word_to_ix)

        ques.append(cls_id + tokenizer.tokenize(curr_question) + sep_id)
        sent.append(tokenizer.tokenize(each_datapoint[3])+ sep_id)
        curr_len = len(ques[ind])+len(sent[ind])
        if(curr_len>max_len):
            max_len = curr_len
        segment_ids.append([0 for x in range(len(ques[ind]))] + [1 for x in range(len(sent[ind]))])
        masking_ids.append([1 for x in range(curr_len)])
        main_drug_ques_ids.append(drug2ix[curr_drug_ques_id.split('-')[0]])
        main_one_hot_vectors.append(curr_one_hot_vector)

    if(max_len > LIMIT_LEN):
        max_len = LIMIT_LEN

    all_lens = []
    for ind in range(len(ques)):
        curr_len = len(ques[ind]) + len(sent[ind])
        tmp_q_sent = tokenizer.convert_tokens_to_ids(ques[ind])+tokenizer.convert_tokens_to_ids(sent[ind])
        ques_sent.append( tmp_q_sent[:max_len] + [0 for x in range(max_len - curr_len)] )
        all_lens.append(len(tmp_q_sent[:max_len]))
        segment_ids[ind] = segment_ids[ind][:max_len] + [1 for x in range(max_len-curr_len)]
        masking_ids[ind] = masking_ids[ind][:max_len] + [0 for x in range(max_len-curr_len)]

    descend_indices = list(np.argsort(all_lens))
    descend_indices.reverse()

    sorted_scores = []
    sorted_sc_weights = []
    sorted_scores.append(class2ix[answer])
    sorted_sc_weights.append(sc_weights[answer])
    sorted_ques_sent = [ques_sent[x] for x in descend_indices]
    sorted_masking_ids = [masking_ids[x] for x in descend_indices]
    sorted_segment_ids = [segment_ids[x] for x in descend_indices]
    sorted_drug_ques_ids = [main_drug_ques_ids[x] for x in descend_indices]
    sorted_lens = [all_lens[x] for x in descend_indices]
    sorted_one_hot_vec = [main_one_hot_vectors[x] for x in descend_indices]
    
    dict_ = {'sents':sorted_ques_sent, 
            'segment_ids':sorted_segment_ids,
            'masking_ids':sorted_masking_ids,
            'scores':sorted_scores,
            'sc_weights':sorted_sc_weights,
            'drug_ques_ids':sorted_drug_ques_ids,
            'lens': sorted_lens,
            'one_hot_vec': sorted_one_hot_vec}
    return dict_

def batchify_bert_classification(all_datapoints, sc_weights, questions, tokenizer, class2ix, LIMIT_LEN = 250):
    ques = []
    sent = []

    ques_sent = []
    segment_ids = []
    masking_ids = []
    labels = []
    label_wts = []
    drug_ques_ids = []

    cls_id = ['[CLS]']
    sep_id = ['[SEP]']
    pad_id = '[PAD]'
    max_len = 0 
    for ind, each_datapoint in enumerate(all_datapoints):
        curr_question = " ".join(questions[each_datapoint[6]])
        ques.append(cls_id + tokenizer.tokenize(curr_question) + sep_id)
        sent.append(tokenizer.tokenize(each_datapoint[3])+ sep_id)
        curr_len = len(ques[ind])+len(sent[ind])
        if(curr_len>max_len):
            max_len = curr_len
        segment_ids.append([0 for x in range(len(ques[ind]))] + [1 for x in range(len(sent[ind]))])
        masking_ids.append([1 for x in range(curr_len)])
        labels.append(class2ix[each_datapoint[5]])
        label_wts.append(sc_weights[each_datapoint[5]])
        drug_ques_ids.append(each_datapoint[6])
    if(max_len > LIMIT_LEN):
        max_len = LIMIT_LEN

    for ind in range(len(ques)):
        curr_len = len(ques[ind]) + len(sent[ind])
        tmp_q_sent = tokenizer.convert_tokens_to_ids(ques[ind])+tokenizer.convert_tokens_to_ids(sent[ind])
        ques_sent.append( tmp_q_sent[:max_len] + [0 for x in range(max_len - curr_len)] )
        segment_ids[ind] = segment_ids[ind][:max_len] + [1 for x in range(max_len-curr_len)]
        masking_ids[ind] = masking_ids[ind][:max_len] + [0 for x in range(max_len-curr_len)]

    dict_ = {'sents':ques_sent, 
            'segment_ids':segment_ids,
            'masking_ids':masking_ids,
            'labels':labels,
            'label_weights':label_wts,
            'drug_ques_ids':drug_ques_ids}
    return dict_

def batchify_bert_classification_para(all_datapoints, sc_weights, questions, tokenizer, class2ix, label_ind=2, LIMIT_LEN = 300):
    ques = []
    sent = []

    ques_sent = []
    segment_ids = []
    masking_ids = []
    labels = []
    label_wts = []
    drug_ques_ids = []

    cls_id = ['[CLS]']
    sep_id = ['[SEP]']
    pad_id = '[PAD]'
    max_len = 0 
    for ind, each_datapoint in enumerate(all_datapoints):
        curr_question = " ".join(questions[each_datapoint[1]])
        ques.append(cls_id + tokenizer.tokenize(curr_question) + sep_id)
        sent.append(tokenizer.tokenize(each_datapoint[0])+ sep_id)
        curr_len = len(ques[ind])+len(sent[ind])
        if(curr_len>max_len):
            max_len = curr_len
        segment_ids.append([0 for x in range(len(ques[ind]))] + [1 for x in range(len(sent[ind]))])
        masking_ids.append([1 for x in range(curr_len)])
        labels.append(class2ix[each_datapoint[label_ind]])
        label_wts.append(sc_weights[each_datapoint[label_ind]])
        drug_ques_ids.append(each_datapoint[1])
    if(max_len > LIMIT_LEN):
        max_len = LIMIT_LEN

    for ind in range(len(ques)):
        curr_len = len(ques[ind]) + len(sent[ind])
        tmp_q_sent = tokenizer.convert_tokens_to_ids(ques[ind])+tokenizer.convert_tokens_to_ids(sent[ind])
        ques_sent.append( tmp_q_sent[:max_len] + [0 for x in range(max_len - curr_len)] )
        segment_ids[ind] = segment_ids[ind][:max_len] + [1 for x in range(max_len-curr_len)]
        masking_ids[ind] = masking_ids[ind][:max_len] + [0 for x in range(max_len-curr_len)]

    dict_ = {'sents':ques_sent, 
            'segment_ids':segment_ids,
            'masking_ids':masking_ids,
            'labels':labels,
            'label_weights':label_wts,
            'drug_ques_ids':drug_ques_ids}
    return dict_

def batchify_bert_classification_para_multitask(all_datapoints, ans_weights, rel_weights, questions, tokenizer, ans2ix, rel2ix, ans_ind=2, rel_ind=3, LIMIT_LEN = 300):
    ques = []
    sent = []

    ques_sent = []
    segment_ids = []
    masking_ids = []
    
    answers = []
    ans_wts = []
    relevances = []
    rel_wts = []
    
    drug_ques_ids = []

    cls_id = ['[CLS]']
    sep_id = ['[SEP]']
    pad_id = '[PAD]'
    max_len = 0 
    for ind, each_datapoint in enumerate(all_datapoints):
        curr_question = " ".join(questions[each_datapoint[1]])
        ques.append(cls_id + tokenizer.tokenize(curr_question) + sep_id)
        sent.append(tokenizer.tokenize(each_datapoint[0])+ sep_id)
        curr_len = len(ques[ind])+len(sent[ind])
        if(curr_len>max_len):
            max_len = curr_len
        segment_ids.append([0 for x in range(len(ques[ind]))] + [1 for x in range(len(sent[ind]))])
        masking_ids.append([1 for x in range(curr_len)])
        
        answers.append(ans2ix[each_datapoint[ans_ind]])
        ans_wts.append(ans_weights[each_datapoint[ans_ind]])

        relevances.append(rel2ix[each_datapoint[rel_ind]])
        rel_wts.append(rel_weights[each_datapoint[rel_ind]])
        drug_ques_ids.append(each_datapoint[1])
    if(max_len > LIMIT_LEN):
        max_len = LIMIT_LEN

    for ind in range(len(ques)):
        curr_len = len(ques[ind]) + len(sent[ind])
        tmp_q_sent = tokenizer.convert_tokens_to_ids(ques[ind])+tokenizer.convert_tokens_to_ids(sent[ind])
        ques_sent.append( tmp_q_sent[:max_len] + [0 for x in range(max_len - curr_len)] )
        segment_ids[ind] = segment_ids[ind][:max_len] + [1 for x in range(max_len-curr_len)]
        masking_ids[ind] = masking_ids[ind][:max_len] + [0 for x in range(max_len-curr_len)]

    dict_ = {'sents':ques_sent, 
            'segment_ids':segment_ids,
            'masking_ids':masking_ids,
            'answers':answers,
            'ans_weights':ans_wts,
            'relevances': relevances, 
            'rel_weights':rel_wts,
            'drug_ques_ids':drug_ques_ids}
    return dict_

def batchify_bert_para(all_datapoints, ans_sc_weights, questions, tokenizer, ans2ix, answer, LIMIT_LEN = 300):
    ques = []
    sent = []

    ques_sent = []
    segment_ids = []
    masking_ids = []
    labels = []
    label_wts = []
    drug_ques_ids = []

    cls_id = ['[CLS]']
    sep_id = ['[SEP]']
    pad_id = '[PAD]'
    max_len = 0 
    for ind, each_datapoint in enumerate(all_datapoints):
        curr_question = " ".join(questions[each_datapoint[1]])
        ques.append(cls_id + tokenizer.tokenize(curr_question) + sep_id)
        sent.append(tokenizer.tokenize(each_datapoint[0])+ sep_id)
        curr_len = len(ques[ind])+len(sent[ind])
        if(curr_len>max_len):
            max_len = curr_len
        segment_ids.append([0 for x in range(len(ques[ind]))] + [1 for x in range(len(sent[ind]))])
        masking_ids.append([1 for x in range(curr_len)])
    if(max_len > LIMIT_LEN):
        max_len = LIMIT_LEN

    for ind in range(len(ques)):
        curr_len = len(ques[ind]) + len(sent[ind])
        tmp_q_sent = tokenizer.convert_tokens_to_ids(ques[ind])+tokenizer.convert_tokens_to_ids(sent[ind])
        ques_sent.append( tmp_q_sent[:max_len] + [0 for x in range(max_len - curr_len)] )
        segment_ids[ind] = segment_ids[ind][:max_len] + [1 for x in range(max_len-curr_len)]
        masking_ids[ind] = masking_ids[ind][:max_len] + [0 for x in range(max_len-curr_len)]

    labels.append(ans2ix[answer])
    label_wts.append(ans_sc_weights[answer])
    drug_ques_ids.append(each_datapoint[1])


    dict_ = {'sents':ques_sent, 
            'segment_ids':segment_ids,
            'masking_ids':masking_ids,
            'labels':labels,
            'label_weights':label_wts,
            'drug_ques_ids':drug_ques_ids}
    return dict_

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def get_start_end_and_tokens(paragraph_text, question_text, orig_answer_text, answer_offset, tokenizer, improve_flag=False):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    start_position = None
    end_position = None
    answer_length = len(orig_answer_text)
    start_position = char_to_word_offset[answer_offset]
    end_position = char_to_word_offset[answer_offset + answer_length - 1]
    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
    cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
    
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if True:
        tok_start_position = orig_to_tok_index[start_position]
        if end_position < len(doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

    if(improve_flag):
        tok_start_position, tok_end_position =  _improve_answer_span(all_doc_tokens, 
                                                                     tok_start_position, 
                                                                     tok_end_position, 
                                                                     tokenizer,
                                                                     orig_answer_text)

    # Adjustment
    tokenized_question = tokenizer.tokenize(question_text)
    tokenized_para = tokenizer.tokenize(paragraph_text)
    tok_start_position = tok_start_position+2+len(tokenized_question) # added 2 for [CLS] and [SEP] token
    tok_end_position = tok_end_position+2+len(tokenized_question)
    return tokenized_question, tokenized_para, tok_start_position, tok_end_position

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def batchify_bert_emrqa(all_datapoints, tokenizer, LIMIT_LEN = 350):
    ques = []
    sent = []

    ques_sent = []
    segment_ids = []
    masking_ids = []
    start_positions = []
    end_positions = []
    drug_ques_ids = []

    cls_id = ['[CLS]']
    sep_id = ['[SEP]']
    pad_id = '[PAD]'
    max_len = 0 
    for ind, each_datapoint in enumerate(all_datapoints):
        datapoint_info = get_start_end_and_tokens(each_datapoint['evidence_sentence'], 
                                                each_datapoint['question'], 
                                                each_datapoint['answer'], 
                                                each_datapoint['answer_start_char_ind'], 
                                                tokenizer)
        ques.append(cls_id+datapoint_info[0]+sep_id)
        sent.append(datapoint_info[1]+sep_id)
        start_positions.append(datapoint_info[2])
        end_positions.append(datapoint_info[3])

        curr_len = len(ques[ind])+len(sent[ind])
        if(curr_len>max_len):
            max_len = curr_len
        segment_ids.append([0 for x in range(len(ques[ind]))] + [1 for x in range(len(sent[ind]))])
        masking_ids.append([1 for x in range(curr_len)])

    if(max_len > LIMIT_LEN):
        max_len = LIMIT_LEN

    for ind in range(len(ques)):
        curr_len = len(ques[ind]) + len(sent[ind])
        tmp_q_sent = tokenizer.convert_tokens_to_ids(ques[ind])+tokenizer.convert_tokens_to_ids(sent[ind])
        ques_sent.append( tmp_q_sent[:max_len] + [0 for x in range(max_len - curr_len)] )
        segment_ids[ind] = segment_ids[ind][:max_len] + [1 for x in range(max_len-curr_len)]
        masking_ids[ind] = masking_ids[ind][:max_len] + [0 for x in range(max_len-curr_len)]

    dict_ = {'sents':ques_sent, 
            'segment_ids':segment_ids,
            'masking_ids':masking_ids,
            'start_positions':start_positions,
            'end_positions':end_positions}
    return dict_

def get_start_end_and_tokens_ernie(paragraph_text, question_text, para_ents, ques_ents, 
                                    orig_answer_text, answer_offset, tokenizer, improve_flag=False):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    start_position = None
    end_position = None
    answer_length = len(orig_answer_text)
    start_position = char_to_word_offset[answer_offset]
    end_position = char_to_word_offset[answer_offset + answer_length - 1]
    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
    cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
    
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens, _ = tokenizer.tokenize(token, [])
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if True:
        tok_start_position = orig_to_tok_index[start_position]
        if end_position < len(doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

    if(improve_flag):
        tok_start_position, tok_end_position =  _improve_answer_span(all_doc_tokens, 
                                                                     tok_start_position, 
                                                                     tok_end_position, 
                                                                     tokenizer,
                                                                     orig_answer_text)

    # Adjustment according to the added CLS and SEP tags
    tokenized_question, tokenized_ques_ents = tokenizer.tokenize(question_text, ques_ents)
    tokenized_para, tokenized_para_ents = tokenizer.tokenize(paragraph_text, para_ents)
    tok_start_position = tok_start_position+2+len(tokenized_question) # added 2 for [CLS] and [SEP] token
    tok_end_position = tok_end_position+2+len(tokenized_question) # added 2 for [CLS] and [SEP] token
    return tokenized_question, tokenized_ques_ents, tokenized_para, tokenized_para_ents, tok_start_position, tok_end_position

def get_entity_ids(entity2id, ents, max_len, curr_len):
    indexed_ents = []
    ent_mask = []
    for ent in ents:
        if ent != "UNK" and ent in entity2id:
            indexed_ents.append(entity2id[ent])
            ent_mask.append(1.0)
        else:
            indexed_ents.append(-1)
            ent_mask.append(0.0) ### We should check this!! IT should be 1 I think
    ent_mask[0] = 1.0 # for CLS as per my understanding

    indexed_ents = indexed_ents[:max_len] + [-1 for i in range(max_len-curr_len)]
    ent_mask = ent_mask[:max_len] + [0.0 for i in range(max_len-curr_len)]

    ## Remember to do +1 for the entities
    indexed_ents = [x+1 for x in indexed_ents]
    return indexed_ents, ent_mask

def batchify_bert_emrqa_ernie(all_datapoints, tokenizer, entity2id, logical2ix,LIMIT_LEN = 350):
    ques = []
    sent = []
    ques_ents = []
    sent_ents = []

    ques_sent = []
    segment_ids = []
    masking_ids = []
    start_positions = []
    end_positions = []
    drug_ques_ids = []
    logical_forms = []

    cls_id = ['[CLS]']
    sep_id = ['[SEP]']
    pad_id = '[PAD]'
    max_len = 0 
    for ind, each_datapoint in enumerate(all_datapoints):
        datapoint_info = get_start_end_and_tokens_ernie(each_datapoint['evidence_sentence'], 
                                                each_datapoint['question'], 
                                                each_datapoint['sent_ents'],
                                                each_datapoint['ques_ents'],
                                                each_datapoint['answer'], 
                                                each_datapoint['answer_start_char_ind'], 
                                                tokenizer)
        ques.append(cls_id+datapoint_info[0]+sep_id)
        ques_ents.append(cls_id+datapoint_info[1]+sep_id)

        sent.append(datapoint_info[2]+sep_id)
        sent_ents.append(datapoint_info[3]+sep_id)

        start_positions.append(datapoint_info[4])
        end_positions.append(datapoint_info[5])
        logical_forms.append(logical2ix[each_datapoint['logical_form_template']])

        curr_len = len(ques[ind])+len(sent[ind])
        if(curr_len>max_len):
            max_len = curr_len
        segment_ids.append([0 for x in range(len(ques[ind]))] + [1 for x in range(len(sent[ind]))])
        masking_ids.append([1 for x in range(curr_len)])

    if(max_len > LIMIT_LEN):
        max_len = LIMIT_LEN

    entity_list  = []
    entity_mask_list = []
    for ind in range(len(ques)):
        curr_len = len(ques[ind]) + len(sent[ind])
        tmp_q_sent = tokenizer.convert_tokens_to_ids(ques[ind])+tokenizer.convert_tokens_to_ids(sent[ind])
        ques_sent.append( tmp_q_sent[:max_len] + [0 for x in range(max_len - curr_len)] )

        tmp_q_sent_ents = ques_ents[ind] + sent_ents[ind]
        entities, ent_mask = get_entity_ids(entity2id, tmp_q_sent_ents, max_len, curr_len)
        entity_list.append(entities)
        entity_mask_list.append(ent_mask)

        segment_ids[ind] = segment_ids[ind][:max_len] + [1 for x in range(max_len-curr_len)]
        masking_ids[ind] = masking_ids[ind][:max_len] + [0 for x in range(max_len-curr_len)]


    dict_ = {'sents':ques_sent, 
            'segment_ids':segment_ids,
            'masking_ids':masking_ids,
            'entities':entity_list,
            'entity_masking_ids':entity_mask_list,
            'start_positions':start_positions,
            'end_positions':end_positions,
            'logical_forms':logical_forms}
    return dict_

def batchify_bert_naranjo_ernie(all_datapoints, tokenizer, entity2id, label2ix, questions,LIMIT_LEN = 500):
    ques = []
    sent = []
    ques_ents = []
    sent_ents = []

    ques_sent = []
    segment_ids = []
    masking_ids = []
    labels = []
    drug_ques_ids = []

    cls_id = ['[CLS]']
    sep_id = ['[SEP]']
    pad_id = '[PAD]'
    max_len = 0 
    for ind, each_datapoint in enumerate(all_datapoints):
        actual_question = " ".join(questions[each_datapoint[1]])

        tmp_q, tmp_q_ent = tokenizer.tokenize(actual_question, each_datapoint[4])
        tmp_ans, tmp_ans_ent = tokenizer.tokenize(each_datapoint[0], each_datapoint[5])

        ques.append(cls_id+tmp_q+sep_id)
        ques_ents.append(cls_id+tmp_q_ent+sep_id)

        sent.append(tmp_ans+sep_id)
        sent_ents.append(tmp_ans_ent+sep_id)

        labels.append(label2ix[each_datapoint[2]])
        drug_ques_ids.append(each_datapoint[1])
        curr_len = len(ques[ind])+len(sent[ind])
        if(curr_len>max_len):
            max_len = curr_len
        segment_ids.append([0 for x in range(len(ques[ind]))] + [1 for x in range(len(sent[ind]))])
        masking_ids.append([1 for x in range(curr_len)])

    if(max_len > LIMIT_LEN):
        max_len = LIMIT_LEN

    entity_list  = []
    entity_mask_list = []
    for ind in range(len(ques)):
        curr_len = len(ques[ind]) + len(sent[ind])
        tmp_q_sent = tokenizer.convert_tokens_to_ids(ques[ind])+tokenizer.convert_tokens_to_ids(sent[ind])
        ques_sent.append( tmp_q_sent[:max_len] + [0 for x in range(max_len - curr_len)] )

        tmp_q_sent_ents = ques_ents[ind] + sent_ents[ind]
        entities, ent_mask = get_entity_ids(entity2id, tmp_q_sent_ents, max_len, curr_len)
        entity_list.append(entities)
        entity_mask_list.append(ent_mask)

        segment_ids[ind] = segment_ids[ind][:max_len] + [1 for x in range(max_len-curr_len)]
        masking_ids[ind] = masking_ids[ind][:max_len] + [0 for x in range(max_len-curr_len)]


    dict_ = {'sents':ques_sent, 
            'segment_ids':segment_ids,
            'masking_ids':masking_ids,
            'entities':entity_list,
            'entity_masking_ids':entity_mask_list,
            'labels':labels,
            'drug_ques_ids':drug_ques_ids}
    return dict_

def batchify_boolQ(all_datapoints, tokenizer, label2ix, entity2id=None, title2ix = None, LIMIT_LEN = 510):
    ques = []
    sent = []
    ques_ents = []
    sent_ents = []

    ques_sent = []
    segment_ids = []
    masking_ids = []
    labels = []
    drug_ques_ids = []
    titles = []

    cls_id = ['[CLS]']
    sep_id = ['[SEP]']
    pad_id = '[PAD]'
    max_len = 0 
    for ind, each_datapoint in enumerate(all_datapoints):
        ques.append(cls_id+tokenizer.tokenize(each_datapoint['question'])+sep_id)
        # ques_ents.append(cls_id+datapoint_info[1]+sep_id)

        sent.append(tokenizer.tokenize(each_datapoint['passage'])+sep_id)
        # sent_ents.append(datapoint_info[3]+sep_id)

        curr_len = len(ques[ind])+len(sent[ind])
        if(curr_len>max_len):
            max_len = curr_len
        segment_ids.append([0 for x in range(len(ques[ind]))] + [1 for x in range(len(sent[ind]))])
        masking_ids.append([1 for x in range(curr_len)])
        curr_answer = str(each_datapoint['answer'])
        labels.append(label2ix[curr_answer])

    if(max_len > LIMIT_LEN):
        max_len = LIMIT_LEN

    # entity_list  = []
    # entity_mask_list = []
    for ind in range(len(ques)):
        curr_len = len(ques[ind]) + len(sent[ind])
        tmp_q_sent = tokenizer.convert_tokens_to_ids(ques[ind])+tokenizer.convert_tokens_to_ids(sent[ind])
        ques_sent.append( tmp_q_sent[:max_len] + [0 for x in range(max_len - curr_len)] )

        # tmp_q_sent_ents = ques_ents[ind] + sent_ents[ind]
        # entities, ent_mask = get_entity_ids(entity2id, tmp_q_sent_ents, max_len, curr_len)
        # entity_list.append(entities)
        # entity_mask_list.append(ent_mask)

        segment_ids[ind] = segment_ids[ind][:max_len] + [1 for x in range(max_len-curr_len)]
        masking_ids[ind] = masking_ids[ind][:max_len] + [0 for x in range(max_len-curr_len)]


    dict_ = {'sents':ques_sent, 
            'segment_ids':segment_ids,
            'masking_ids':masking_ids,
            'entities':None,
            'entity_masking_ids':None,
            'labels':labels,
            'titles':titles}
    return dict_

def batchify_boolQ_ernie(all_datapoints, tokenizer, label2ix, entity2id=None, title2ix = None, LIMIT_LEN = 510):
    ques = []
    sent = []
    ques_ents = []
    sent_ents = []

    ques_sent = []
    segment_ids = []
    masking_ids = []
    labels = []
    drug_ques_ids = []
    titles = []
    all_lens = []

    cls_id = ['[CLS]']
    sep_id = ['[SEP]']
    pad_id = '[PAD]'
    max_len = 0 
    for ind, each_datapoint in enumerate(all_datapoints):

        tmp_q, tmp_q_ent = tokenizer.tokenize(each_datapoint['question'], each_datapoint['ques_ents'])
        ques.append(cls_id+tmp_q+sep_id)
        ques_ents.append(cls_id+tmp_q_ent+sep_id)

        tmp_s, tmp_s_ent = tokenizer.tokenize(each_datapoint['passage'], each_datapoint['passage_ents'])
        sent.append(tmp_s+sep_id)
        sent_ents.append(tmp_s_ent+sep_id)

        curr_len = len(ques[ind])+len(sent[ind])
        if(curr_len>max_len):
            max_len = curr_len
        segment_ids.append([0 for x in range(len(ques[ind]))] + [1 for x in range(len(sent[ind]))])
        masking_ids.append([1 for x in range(curr_len)])
        curr_answer = str(each_datapoint['answer'])
        labels.append(label2ix[curr_answer])
        titles.append(title2ix[each_datapoint['title_label']])

    if(max_len > LIMIT_LEN):
        max_len = LIMIT_LEN

    entity_list  = []
    entity_mask_list = []
    for ind in range(len(ques)):
        curr_len = len(ques[ind]) + len(sent[ind])
        tmp_q_sent = tokenizer.convert_tokens_to_ids(ques[ind])+tokenizer.convert_tokens_to_ids(sent[ind])
        ques_sent.append( tmp_q_sent[:max_len] + [0 for x in range(max_len - curr_len)] )
        all_lens.append(len(tmp_q_sent[:max_len]))

        tmp_q_sent_ents = ques_ents[ind] + sent_ents[ind]
        entities, ent_mask = get_entity_ids(entity2id, tmp_q_sent_ents, max_len, curr_len)
        entity_list.append(entities)
        entity_mask_list.append(ent_mask)

        segment_ids[ind] = segment_ids[ind][:max_len] + [1 for x in range(max_len-curr_len)]
        masking_ids[ind] = masking_ids[ind][:max_len] + [0 for x in range(max_len-curr_len)]


    dict_ = {'sents':ques_sent, 
            'segment_ids':segment_ids,
            'masking_ids':masking_ids,
            'entities':entity_list,
            'entity_masking_ids':entity_mask_list,
            'labels':labels,
            'titles':titles,
            'lens':all_lens}
    return dict_

############################# GET METRICS #######################################

def get_metrics(actual_outputs, preds):
    df_prf_classwise = get_prf_classwise(actual_outputs, preds)
    weighted_prf = precision_recall_fscore_support(actual_outputs, preds, average='weighted')
    macro_prf = precision_recall_fscore_support(actual_outputs, preds, average='macro')
    micro_prf = precision_recall_fscore_support(actual_outputs, preds, average='micro')

    all_prfs = list(weighted_prf)[:-1]+list(macro_prf)[:-1]+list(micro_prf)[:-1]
    df_prfs = pd.DataFrame(all_prfs) #, columns = ['w-p','w-r','w-f','ma-p','ma-r','ma-f','mi-p','mi-r','mi-f'])
    df_prfs = df_prfs.T
    df_prfs.columns = ['w-p','w-r','w-f','ma-p','ma-r','ma-f','mi-p','mi-r','mi-f']

    return df_prfs, df_prf_classwise

def get_prf_classwise(act_, pred_):
    act_unique = list(set(act_))
    prf = precision_recall_fscore_support(act_,pred_,labels=act_unique)
    prf = [x for x in prf]
    prf = pd.DataFrame(prf, columns= act_unique, index=['precision','recall','f-score','Count'])
    return prf

def get_question_wise_metrics(actual_rel, pred_rel, questionIds):
    qs = [2,3,5,7,10]
    dict_results = {}
    for each_q in qs:
        q_key = str(each_q)
        # relevance
        tmp_act = [actual_rel[ind] for ind in range(len(questionIds)) if(q_key in questionIds[ind])]
        tmp_pred = [pred_rel[ind] for ind in range(len(questionIds)) if(q_key in questionIds[ind])]

        tmp_rel_metrics = get_metrics(actual_outputs=tmp_act, preds=tmp_pred)
        
        dict_results[q_key] = tmp_rel_metrics
    return dict_results

######################## Conversions #############################################

def convert_rel_score_to_one_class(rel, score):
    combined_ = []
    for ind in range(len(rel)):
        if(rel[ind]==0):
            combined_.append(3)
        else:
            combined_.append(score[ind])
    return combined_