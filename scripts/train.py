import pickle
from random import shuffle
from tqdm import tqdm
import torch
import torch.nn as nn

import utils as utils
import eval_plot as eval_plot
import argparse

def train(args, model, optimizer, tokenizer, model_config, data, entity2id, logical2ix):

    num_train_epochs = args.train_epochs
    warmup_proportion = 0.1

    device = torch.device("cuda:"+str(args.gpu))
    model = torch.nn.DataParallel(model, device_ids=[args.gpu])
    model.to(device)

    train_losses = []
    train_acc = []

    batch_size = model_config['batch_size']
    train_data = data['train']
    loss_lf_fct = nn.CrossEntropyLoss()

    for epoch in range(num_train_epochs):
        epoch_key = "Epoch_" + str(epoch)
        print('==' * 22 + epoch_key + "==" * 22)
        train_loss = 0
        model.train()
        for ind in tqdm(range(0, len(train_data), batch_size)):
            if args.model_type == 'bert':
                current_batch = utils.batchify_bert_emrqa(train_data[ind:ind + batch_size],
                                                          tokenizer)
            elif args.model_type == 'ernie':
                current_batch = utils.batchify_bert_emrqa_ernie(train_data[ind:ind + batch_size],
                                                                tokenizer,
                                                                entity2id,
                                                                logical2ix)
                all_ent_ids = torch.tensor(current_batch['entities'], dtype=torch.long)
                all_ent_masks = torch.tensor(current_batch['entity_masking_ids'], dtype=torch.long)
                all_logical_forms = torch.tensor(current_batch['logical_forms'], dtype=torch.long).to(device)

            all_input_ids = torch.tensor(current_batch['sents'], dtype=torch.long)
            all_input_mask = torch.tensor(current_batch['masking_ids'], dtype=torch.long)

            all_segment_ids = torch.tensor(current_batch['segment_ids'], dtype=torch.long)
            all_start_positions = torch.tensor(current_batch['start_positions'], dtype=torch.long)
            all_end_positions = torch.tensor(current_batch['end_positions'], dtype=torch.long)

            if args.model_type == 'bert':
                loss = model(all_input_ids, all_segment_ids, all_input_mask, all_start_positions, all_end_positions)
            elif args.model_type == 'ernie':
                loss, lf_logits = model(input_ids=all_input_ids,
                                        token_type_ids=all_segment_ids,
                                        attention_mask=all_input_mask,
                                        input_ents_idx=all_ent_ids,
                                        ent_mask=all_ent_masks,
                                        start_positions=all_start_positions,
                                        end_positions=all_end_positions)
            if args.multi_task and args.model_type=='ernie':
                # if multi-task learning is activated, incorporate the loss of
                loss_lf = loss_lf_fct(lf_logits, all_logical_forms)
                loss = loss + loss_lf


            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss)
        torch.save(model, '../models/' + model_config['model_name'] + epoch_key + '.pt')
        print('--' * 50)
        print('--' * 50)
    torch.save(model, '../models/' + model_config['model_name'] + '_final.pt')
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Type of model to train.",
    )
    parser.add_argument(
        "--model_save_name",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--train_setting",
        default='relaxed',
        type=str,
        required=False,
        help="Whether to train in strict setting or relaxed setting. Options: strict or relaxed",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run the model on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run the model on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="Whether to evaluate during training.")
    parser.add_argument("--multi_task", action="store_true", help="Multi-task learning flag.")

    parser.add_argument(
        "--train_batch_size", default=20, type=int, help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--train_epochs", default=5, type=int, help="Training epochs."
    )
    parser.add_argument(
        "--GRAD_ACC", default=1, type=int, help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--eval_batch_size", default=20, type=int, help="Batch size per GPU/CPU for evaluation/testing."
    )
    parser.add_argument(
        "--lr", default=2e-5, type=float, help="Learning rate."
    )
    parser.add_argument(
        "--auxiliary_task_wt", default=0.3, type=float, help="Weight for the auxiliary task."
    )
    parser.add_argument(
        "--weight_decay", default=1e-4, type=float, help="Weight decay."
    )
    parser.add_argument(
        "--warmup_proportion", default=0.1, type=float, help="Warmup proportion."
    )
    parser.add_argument(
        "--gpu", default=0, type=int, help="which GPU is to be used for training."
    )

    args = parser.parse_args()

    data = pickle.load(open(args.data_dir, 'rb'))
    selected_sem_types = pickle.load(open('../data/selected_ents.pkl', 'rb'))
    print('Selected semantic types: ', selected_sem_types)

    if args.train_setting == 'strict':
        data = data['strict_split']
    else:
        data = data['split']

    entity2id = utils.prepare_entities_to_ix(selected_sem_types)
    logical2ix = utils.prepare_logical_forms_to_ix(data['train'])

    shuffle(data['train'])
    shuffle(data['dev'])
    shuffle(data['test'])
    print(entity2id)

    model_config = {
        'label_size': 2,
        'num_entities': len(selected_sem_types) + 1,
        'entity_dim': 100,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.train_batch_size,
        'data_path': args.data_dir,
        'model_name': args.model_save_name,
        'bert_model': args.model_name_or_path,
        'do_lower_case': True,
        'gradient_accumulation_steps': args.GRAD_ACC
    }

    if args.model_type == 'ernie':
        from knowledge_bert import modeling
        from knowledge_bert import BertTokenizer
        from knowledge_bert.optimization import BertAdam

        tokenizer = BertTokenizer.from_pretrained(model_config['bert_model'],
                                                  do_lower_case=model_config['do_lower_case'])
        model, _ = modeling.BertForQuestionAnsweringEmrQA.from_pretrained(model_config['bert_model'],
                                                                      num_entities=model_config['num_entities'])
    elif args.model_type == 'bert':
        from pytorch_pretrained_bert import BertTokenizer, BertForQuestionAnswering
        from pytorch_pretrained_bert.optimization import BertAdam
        tokenizer = BertTokenizer.from_pretrained(model_config['bert_model'],
                                                  do_lower_case=model_config['do_lower_case'])
        model = BertForQuestionAnswering.from_pretrained(model_config['bert_model'])

    num_train_optimization_steps = len(data['train']) // model_config[
        'gradient_accumulation_steps'] * args.train_epochs

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=model_config['lr'],
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    if args.do_train:
        model_trained = train(args, model=model,
                              optimizer=optimizer, tokenizer=tokenizer, model_config=model_config,
                              data=data, entity2id=entity2id, logical2ix=logical2ix)

    # The start and end accuracy are just proxies, actual accuracy would be calculated from the pickle dump using the script of SQuAD evaluate: https://rajpurkar.github.io/SQuAD-explorer/
    ##### Evaluate the model if do_eval flag is on
    if args.do_eval:
        if args.model_type == 'ernie':
            if args.multi_task:
                device = torch.device("cuda:" + str(args.gpu))
                dev_vals = eval_plot.evaluate_bert_emrqa_ernie_multitask(model_trained,
                                                                         data['dev'],
                                                                         args.eval_batch_size,
                                                                         tokenizer,
                                                                         entity2id,
                                                                         logical2ix,
                                                                         device)
            else:
                dev_vals = eval_plot.evaluate_bert_emrqa_ernie(model_trained,
                                                       data['dev'],
                                                       args.eval_batch_size,
                                                       tokenizer,
                                                       entity2id,
                                                       logical2ix)
        elif args.model_type == 'bert':
            dev_vals = eval_plot.evaluate_bert_emrqa(model_trained,
                                                     data['dev'],
                                                     args.eval_batch_size,
                                                     tokenizer)
        dict_ = {'start_accuracy': dev_vals[0],
                 'end_accuracy': dev_vals[1],
                 'actual_and_predicted_values': dev_vals[2]}
        file_name = '../results/'+model_config['model_name']+'_dev_results.pkl'
        pickle.dump(dict_, open(file_name, 'wb'))

    ##### Test the model
    if args.do_test:
        if args.model_type == 'ernie':
            if args.multi_task:
                device = torch.device("cuda:" + str(args.gpu))
                test_vals = eval_plot.evaluate_bert_emrqa_ernie_multitask(model_trained,
                                                                         data['test'],
                                                                         args.eval_batch_size,
                                                                         tokenizer,
                                                                         entity2id,
                                                                         logical2ix,
                                                                         device)
            else:
                test_vals = eval_plot.evaluate_bert_emrqa_ernie(model_trained,
                                                       data['test'],
                                                       args.eval_batch_size,
                                                       tokenizer,
                                                       entity2id,
                                                       logical2ix)
        elif args.model_type == 'bert':
            test_vals = eval_plot.evaluate_bert_emrqa(model_trained,
                                                     data['dev'],
                                                     args.eval_batch_size,
                                                     tokenizer)
        dict_ = {'start_accuracy': test_vals[0],
                     'end_accuracy': test_vals[1],
                     'actual_and_predicted_values': test_vals[2]}
        file_name = '../results/' + model_config['model_name'] + '_test_results.pkl'
        pickle.dump(dict_, open(file_name, 'wb'))

if __name__ == "__main__":
    main()