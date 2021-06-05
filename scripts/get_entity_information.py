from pymetamap import MetaMap
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm_notebook
import pickle

# Check the metamap instance is working.
mm = MetaMap.get_instance('/home/bsingh/public_mm/bin/metamap16')
print(mm.extract_concepts(['Heart Attack', 'I am having a heart attack',], [0,1]))


def get_semantic_types_dict(list_, curr_list_concepts, sel_sem_types):
    dict_con = {}
    failed_con = 0
    for ind in range(len(list_)):
        dict_con[ind] = []
    for each_con in tqdm(curr_list_concepts):
        try:
            pos_info = each_con.pos_info
            pos_info = pos_info.replace('[','')
            pos_info = pos_info.replace(']','')
            if(';' in pos_info):
                all_pos = pos_info.split(';')
            else:
                all_pos = pos_info.split(',')
            all_pos_final = []
            for each_pos in all_pos:
                all_pos_final+=each_pos.split(',')
            for each_pos in all_pos_final:
                st = int(each_pos.split('/')[0])-1
                end = st+int(each_pos.split('/')[1])
                semtype_all = each_con.semtypes[1:-1].split(',')
                semtype_all = [x for x in semtype_all if(x in sel_sem_types)]
                # can be handled better
                if(len(semtype_all)>0):
                    semtype = semtype_all[0]
                else:
                    semtype = 'Ignore'
                if(semtype in sel_sem_types):
                    dict_con[int(each_con.index)].append([semtype, st, end, 1.0])
        except:
            failed_con+=1
    print("Total_concepts", curr_list_concepts.__len__())
    print("Failed_concepts", failed_con)
    final_dict = {}
    for ind in range(len(list_)):
        final_dict[list_[ind]] = dict_con[ind]
    return final_dict

def get_all_concepts(curr_mm, all_sents, batch_size=10):
    all_concepts = []
    for i in tqdm_notebook(range(0, all_sents.__len__(), batch_size)):
        try:
            curr_sents = all_sents[i:i+batch_size]
            tmp = curr_mm.extract_concepts(curr_sents, [x for x in range(i, i+curr_sents.__len__())])
            all_concepts.append(tmp)
        except: 
            print("problem with the batch starting from"+str(i))
    return all_concepts

def collate_all_concepts(tmp_concepts):
    final_concepts = []
    for each_concept in tmp_concepts:
        final_concepts+=each_concept[0]
    return final_concepts

data = pickle.load(open('../data/emrqa_parawise_data.pkl','rb'))
# As we need the semantic types for the questions as well as paras from both splits
all_data = data['strict_split']['train']+data['strict_split']['dev']+data['strict_split']['test']
all_data = all_data+data['split']['train']+data['split']['dev']+data['split']['test']

# Here the evidence_sentence refers to the paragraphs extracted earlier.
all_sents = [x['evidence_sentence'] for x in all_data]
all_ques = [x['question'] for x in all_data]

all_sents = list(set(all_sents))
all_ques = list(set(all_ques))

selected_sem_types = open('../data/selected_semantic_types.txt','r').read()
selected_sem_types = selected_sem_types.split('\n')
selected_sem_types = [x.split(':')[0].strip() for x in selected_sem_types]

print('Selected semantic types:', selected_sem_types)


all_emrqa_sents_con = get_all_concepts(all_sents=all_sents)
all_emrqa_ques_con = get_all_concepts(all_sents=all_ques)

final_emrqa_sents_con = collate_all_concepts(all_emrqa_sents_con)
dict_con_new_sent = get_semantic_types_dict(all_sents, final_emrqa_sents_con, selected_sem_types)

final_emrqa_ques_con = collate_all_concepts(all_emrqa_ques_con)
dict_con_new_ques = get_semantic_types_dict(all_ques, final_emrqa_ques_con, selected_sem_types)


# ## Dumping the metamap annotations
# ---------------------
pickle.dump((dict_con_new_ques, dict_con_new_sent),
            open('../data/metamap_annotations_emrqa_dict.pkl','wb'))
pickle.dump(selected_sem_types, open('../data/selected_ents.pkl','wb'))


# ## Updating the data with the semantic type infomrmation
# ----------------------------------

concept_dict = pickle.load(open('../data/metamap_annotations_emrqa_dict_all.pkl', 'rb'))

ques_concept_dict = concept_dict[0]
sent_concept_dict = concept_dict[1]

def add_entity_information(list_, dict_ques, dict_sents):
    for each_ in tqdm(list_):
        each_['sent_ents'] = dict_sents[each_['evidence_sentence']]
        each_['ques_ents'] = dict_ques[each_['question']]
    print('Done.')

add_entity_information(data['split']['train'], ques_concept_dict, sent_concept_dict)

print('---'*10)
print('Updated Datapoint Example')
print('---'*10)
print(data['split']['train'][0])


add_entity_information(data['split']['dev'], ques_concept_dict, sent_concept_dict)
add_entity_information(data['split']['test'], ques_concept_dict, sent_concept_dict)


### Strict Split ###
add_entity_information(data['strict_split']['train'], ques_concept_dict, sent_concept_dict)
add_entity_information(data['strict_split']['dev'], ques_concept_dict, sent_concept_dict)
add_entity_information(data['strict_split']['test'], ques_concept_dict, sent_concept_dict)



pickle.dump(data, open('../data/emrqa_parawise_data_w_entities.pkl', 'wb'))