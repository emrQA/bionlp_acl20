import pandas as pd
import json
from tqdm import notebook
from random import shuffle
from collections import Counter
import pickle
import numpy.random as random
from ipdb import set_trace as breakpoint
from nltk.tokenize import sent_tokenize

def read_json(file_name):
    with open(file_name) as f:
        dataset_json = json.load(f)
    return dataset_json


medication_json = read_json('../data/medication-qa.json')
relations_json = read_json('../data/relations-qa.json')
risk_json = read_json('../risk-qa.json')

risk_csv = pd.read_csv('../data/risk-ql.csv', sep="\t")
relations_csv = pd.read_csv('../data/relations-ql.csv', sep='\t')
medication_csv = pd.read_csv('../data/medication-ql.csv', sep='\t')

print(medication_json['paragraphs'][0]['qas'][0]['question'])


sum([x['qas'].__len__() for x in medication_json['paragraphs']])

def get_question_logical_form_mapping(curr_csv):
    dict_mapping = {}
    indices = curr_csv.index
    for each_ind in indices:
        if(each_ind[0] not in dict_mapping):
            dict_mapping[each_ind[0]] = each_ind[1]
    return dict_mapping

def get_start_end_token(evidence, answer):
    start_char_ind = evidence.lower().find(answer.lower())
    end_char_ind = start_char_ind+len(answer)
    return [evidence, answer, start_char_ind, end_char_ind]

def populate_for_all_questions(dataset, curr_ans_list, list_ques, main_logical_form, note_id, data_json_mapping):
    return [[dataset, note_id, main_logical_form]+[data_json_mapping[x[0]], x[1], x[0]]+curr_ans_list for x in list_ques]

def get_context(sentence, context, PARA_size=10):
    orig_sentence = sentence
    sentence = sentence.lower()
#     sentence = sentence.replace('\n', '')
    orig_context= context
    context = [' '.join(x.split()) for x in context]
    context = [x.strip().lower() for x in context]
    medical_note = ' '.join(context)
    medical_note = ' '.join(medical_note.split())
    if sentence not in medical_note:
        test_dp['note'] = medical_note
        test_dp['orig_note'] = orig_context
        test_dp['search_sentence'] = sentence
        print('The answer somehow is not in the actual evidence.')
    
    start_sentence_idx = medical_note.find(sentence)
    end_sentence_idx = start_sentence_idx + len(sentence)
    before_part_note = medical_note[:start_sentence_idx]
    later_part_note = medical_note[end_sentence_idx:]
    before_sents_cnt = random.randint(0,PARA_size)
    
    before_sents = sent_tokenize(before_part_note)
    later_sents = sent_tokenize(later_part_note)
    
    if len(before_sents)<before_sents_cnt:
        before_sents_cnt = len(before_sents)
    
    rem_sents_cnt = PARA_size - before_sents_cnt
    final_para = before_sents[-before_sents_cnt:] + [sentence] + later_sents[:rem_sents_cnt]
    final_para = ' '.join(final_para)
    final_para = ' '.join(final_para.split())
    
    return final_para
    
    flag_not_found = False
    for ind, val in enumerate(context):
        if(sentence in val):
            flag_not_found = True
            break
#     if flag_not_found:
    if True:
        chosen_sent_ind = ind
        before_sents = random.randint(0,PARA_size)
        start_ind = chosen_sent_ind-before_sents
        if(start_ind<0):
            start_ind = 0
        rem = PARA_size - before_sents
        end_ind = chosen_sent_ind+rem+1

        para = context[start_ind:end_ind]
        para = " ".join(para)
    else:
        test_dp['sentence'] = sentence
        test_dp['context'] = context
        test_dp['orig_sentence'] = orig_sentence
        print('Error: the sentence not found')
        exit()
    return para
    
def get_qas_list(data_json, data_json_mapping, dataset=None, PARA_size=10):
    all_data_list = []
    cnt_yn = 0
    cnt_ma = 0
    total_size = 0
    cnt_not_found = 0
    cnt_answer_evidence_sentence_mismatch = 0
    rejected_list_yes_no = []
    rejected_list_multiple_answers = []
    for each_para in notebook.tqdm(data_json['paragraphs']):
        note_id = each_para['note_id']
        cnt_w = -1
        for each_qas in each_para['qas']:
            cnt_w+=1
            list_ques = each_qas['id'][0]
            main_logical_form = each_qas['id'][1]
            for each_ans in each_qas['answers']:
                total_size+=1
                answer_phrase = each_ans['text']
                answer_evidence = each_ans['evidence']
                if(answer_phrase!=""):
                    if(str(type(answer_evidence))=="<class 'str'>"):
                        answer_phrase = ' '.join(answer_phrase.split())
                        answer_evidence = ' '.join(answer_evidence.split())
                        answer_found = True if answer_phrase.lower() in answer_evidence.lower() else False
                        test_dp['answer_phrase'] = each_ans['text']
                        test_dp['answer_evidence'] = each_ans['evidence']
                        test_dp['context'] = each_para['context']
                        # This condition is when something is wrong and the answer is not matching in the original evidence sentence.
                        if not answer_found:
                            cnt_answer_evidence_sentence_mismatch+=1
                            continue
                        answer_evidence = get_context(answer_evidence, each_para['context'], PARA_size)
                        curr_ans_list = get_start_end_token(answer_evidence, answer_phrase)
                        if(curr_ans_list[2]!=-1):
                            curr_ques_ans_list = populate_for_all_questions(dataset,
                                                                            curr_ans_list, 
                                                                            list_ques, 
                                                                            main_logical_form,
                                                                            note_id,
                                                                            data_json_mapping)
                            all_data_list+=curr_ques_ans_list
                        else:
                            test_dp['answer_evidence'] = answer_evidence
                            test_dp['answer_phrase'] = answer_phrase
                            cnt_not_found+=1
                    else:
                        cnt_ma+=1
                        rejected_list_multiple_answers.append(each_ans)
                else:
                    rejected_list_yes_no.append(each_ans)
                    cnt_yn+=1
    print('Dataset: ', dataset)                
    print('Percentage Rejected Yes No: ', cnt_yn*100.0/total_size)
    print('Percentage Rejected MA: ', cnt_ma*100.0/total_size)
    print('Percentage Not Found: ', cnt_not_found*100.0/total_size)
    print('Percentage evidence phrase sentence mismatch: ', cnt_answer_evidence_sentence_mismatch*100.0/total_size)
    print('Total size: ', total_size)
    print('Extracted size: ', len(all_data_list))
    print('--'*20)
    return all_data_list, (rejected_list_yes_no, rejected_list_multiple_answers)

medication_ques_mapping = get_question_logical_form_mapping(medication_csv)
risk_ques_mapping = get_question_logical_form_mapping(risk_csv)
relations_ques_mapping = get_question_logical_form_mapping(relations_csv)


risk_list, rej_risk = get_qas_list(risk_json, risk_ques_mapping, 'risk')


medication_list, rej_med = get_qas_list(medication_json, medication_ques_mapping, 'medication')

relations_list, rej_rel = get_qas_list(relations_json, relations_ques_mapping, 'relations')



print('--'*30)
print(len(medication_list))
print(len(risk_list))
print(len(relations_list))
print('--'*30)
print(Counter([x['answer_entity_type'] for x in rej_med[0]]))
print(Counter([x['answer_entity_type'] for x in rej_risk[0]]))
print(Counter([x['answer_entity_type'] for x in rej_rel[0]]))
print('--'*30)
print(Counter([x['answer_entity_type'] for x in rej_med[1]]))
print(Counter([x['answer_entity_type'] for x in rej_risk[1]]))
print(Counter([x['answer_entity_type'] for x in rej_rel[1]]))


def convert_list_to_dict(list_):
    final_list = []
    for each_ in list_:
        dict_ = {}
        dict_['dataset'] = each_[0]
        dict_['note_id'] = each_[1]
        dict_['logical_form_template'] = each_[2]
        dict_['logical_form'] = each_[3]
        dict_['question_template'] = each_[4]
        dict_['question'] = each_[5]
        dict_['evidence_sentence'] = each_[6]
        dict_['answer'] = each_[7]
        dict_['answer_start_char_ind'] = each_[8]
        dict_['answer_end_char_ind'] = each_[9]
        final_list.append(dict_)
    shuffle(final_list)
    return final_list



medication_list = convert_list_to_dict(medication_list)
risk_list = convert_list_to_dict(risk_list)
relations_list = convert_list_to_dict(relations_list)


print(medication_list[1])


def get_logical_form_distr(list_):
    all_logical_forms = list(set([x['logical_form_template'] for x in list_]))
    print('Total LFs:', len(all_logical_forms))
    total_question_templates = 0
    train_lf_set = []
    dev_lf_set = []
    test_lf_set = []
    for each_logical_form in all_logical_forms:
        print(each_logical_form)
        all_question_templates = [x['question_template'] for x in list_ if(x['logical_form_template']==each_logical_form)]
        all_question_templates = list(set(all_question_templates))
        total_question_templates+=len(all_question_templates)
        print("No of question templates: ", len(all_question_templates))
        if(len(all_question_templates)==1):
            train_lf_set+=all_question_templates
        elif(len(all_question_templates)==2):
            train_lf_set+=all_question_templates[:1]
            dev_lf_set+=all_question_templates[1:]
            test_lf_set+=all_question_templates[1:]
        else:
            train_lf_set+=all_question_templates[:int(len(all_question_templates)*0.70)]
            dev_lf_set+=all_question_templates[int(len(all_question_templates)*0.70):]
            test_lf_set+=all_question_templates[int(len(all_question_templates)*0.70):]
        print('--'*20)
    print('=='*20)
    print('Summary')
    print('=='*20)
    print("Total question templates: ", total_question_templates)
    chk = {'train_set':train_lf_set, 'dev_set':dev_lf_set, 'test_set':test_lf_set}
    print('Train:', len(chk['train_set']))
    print('Development:', len(chk['dev_set']))
    print('Test:', len(chk['test_set']))
    print('Sanity check, train and dev:', len(chk['train_set'])+len(chk['dev_set']))
    print('Sanity check, train and test:', len(chk['train_set'])+len(chk['test_set']))
    print('Sanity check, intersection:', set(chk['train_set']).intersection(set(chk['dev_set'])).__len__())
    print('Sanity check, intersection:', set(chk['train_set']).intersection(set(chk['test_set'])).__len__())
    print('Sanity check, intersection:', set(chk['test_set']).intersection(set(chk['dev_set'])).__len__())
    return chk


dict_question_template = get_logical_form_distr(medication_list+risk_list+relations_list)



def divide_train_dev_test(list_, dict_):
    total_notes = list(set([x['note_id'] for x in list_]))
    train_notes = total_notes[:int(0.60*len(total_notes))]
    dev_notes = total_notes[int(0.60*len(total_notes)):int(0.80*len(total_notes))]
    test_notes = total_notes[int(0.80*len(total_notes)):]
    
    train_data = [x for x in list_ if((x['note_id'] in train_notes) 
                                      and (x['question_template'] in dict_['train_set']))]
    dev_data = [x for x in list_ if((x['note_id'] in dev_notes) 
                                      and (x['question_template'] in dict_['dev_set']))]
    test_data = [x for x in list_ if((x['note_id'] in test_notes) 
                                      and (x['question_template'] in dict_['test_set']))]
    
    train_data_all_temps = [x for x in list_ if(x['note_id'] in train_notes)]
    dev_data_all_temps = [x for x in list_ if(x['note_id'] in dev_notes)]
    test_data_all_temps = [x for x in list_ if(x['note_id'] in test_notes)]
    
    print('=='*20)
    print('Strict Split')
    print('=='*20)
    final_data = {}
    final_data['strict_split']={}
    final_data['strict_split']['train'] = train_data
    final_data['strict_split']['dev'] = dev_data
    final_data['strict_split']['test'] = test_data
    print('Strict split train: ', len(train_data))
    print('Strict split dev: ', len(dev_data))
    print('Strict split test: ', len(test_data))
    print('Total Strict split: ', len(train_data)+len(dev_data)+len(test_data))
    
    print('=='*20)
    print('Normal Split')
    print('=='*20)
    final_data['split'] = {}
    final_data['split']['train'] = train_data_all_temps
    final_data['split']['dev'] = dev_data_all_temps
    final_data['split']['test'] = test_data_all_temps
    print('Split train: ', len(train_data_all_temps))
    print('Split dev: ', len(dev_data_all_temps))
    print('Split test: ', len(test_data_all_temps))
    print('Total split: ', len(train_data_all_temps)+len(dev_data_all_temps)+len(test_data_all_temps))
    
    return final_data

data_split = divide_train_dev_test(medication_list+risk_list+relations_list, dict_question_template)


def check_overlaps(dict_):
    train_set_lft = [x['logical_form_template'] for x in dict_['train']]
    dev_set_lft = [x['logical_form_template'] for x in dict_['dev']]
    test_set_lft = [x['logical_form_template'] for x in dict_['test']]
    
    train_set_qt = [x['question_template'] for x in dict_['train']]
    dev_set_qt = [x['question_template'] for x in dict_['dev']]
    test_set_qt = [x['question_template'] for x in dict_['test']]
    
    print('Num of logical forms in Train:', len(set(train_set_lft)))
    print('Num of logical forms in Dev:', len(set(dev_set_lft)))
    print('Num of logical forms in Test:', len(set(test_set_lft)))
    print('--'*20)
    print("Train-Dev LF-template Overlap", len(set(train_set_lft).intersection(set(dev_set_lft))))
    print("Train-Test LF-template Overlap", len(set(train_set_lft).intersection(set(test_set_lft))))
    print("Test-Dev LF-template Overlap", len(set(test_set_lft).intersection(set(dev_set_lft))))
    print('--'*20)
    print('Num of question templates in Train:', len(set(train_set_qt)))
    print('Num of question templates in Dev:', len(set(dev_set_qt)))
    print('Num of question templates in Test:', len(set(test_set_qt)))
    print('--'*20)
    print("Train-Dev Ques-template Overlap", len(set(train_set_qt).intersection(set(dev_set_qt))))
    print("Train-Test Ques-template Overlap", len(set(train_set_qt).intersection(set(test_set_qt))))
    print("Test-Dev Ques-template Overlap", len(set(test_set_qt).intersection(set(dev_set_qt))))


check_overlaps(data_split['strict_split'])


check_overlaps(data_split['split'])


pickle.dump(data_split, open('../data/emrqa_parawise_data.pkl','wb'))


print(data_split['strict_split']['train'][0])