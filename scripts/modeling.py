from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import CrossEntropyLoss


class BertForQA_lf(BertPreTrainedModel):
    def __init__(self, config, num_ents=10, num_lfs=10):
        super(BertForQA_lf, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.ent_outputs = nn.Linear(config.hidden_size, num_ents)
        self.lf_outputs = nn.Linear(config.hidden_size, num_lfs)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,
                end_positions=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask,
                                                       output_all_encoded_layers=False)
        sequence_output, attn_op = outputs
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        ent_logits = self.ent_outputs(sequence_output)
        lf_logits = self.lf_outputs(attn_op)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        return start_logits, end_logits, ent_logits, lf_logits

class BertForQA(BertPreTrainedModel):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False, num_ents=10, num_lfs=10):
        super(BertForQA, self).__init__(config)
        self.output_attentions = output_attentions
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.ent_outputs = nn.Linear(config.hidden_size, num_ents)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,
                end_positions=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask,
                                                       output_all_encoded_layers=False)
        sequence_output, _ = outputs
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        ent_logits = self.ent_outputs(sequence_output)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        return start_logits, end_logits, ent_logits