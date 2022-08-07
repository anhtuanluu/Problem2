import torch
from torch import nn
from transformers import *

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    'roberta-base-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    'roberta-large-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}

# class BertForQNHackathon(BertPreTrainedModel):
#     config_class = RobertaConfig
#     pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
#     base_model_prefix = "roberta"
#     def __init__(self, config):
#         super(BertForQNHackathon, self).__init__(config)
#         self.num_labels = config.num_labels
#         self.phobert = AutoModel.from_pretrained("vinai/phobert-base", config=config)
#         self.qa_outputs = nn.Linear(4 * config.hidden_size, self.num_labels)
#         self.init_weights()

#     def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
#                         start_positions=None, end_positions=None):
#         outputs = self.phobert(input_ids,
#                             attention_mask=attention_mask,
#                             position_ids=position_ids,
#                             head_mask=head_mask)
#         # print(outputs[2][-1].shape)
#         # outputs = outputs.hidden_states
#         cls_output = torch.cat((outputs[2][-1][:,0, ...],
#                                 outputs[2][-2][:,0, ...],
#                                 outputs[2][-3][:,0, ...],
#                                 outputs[2][-4][:,0, ...]), 
#                                 -1)
#         logits = self.qa_outputs(cls_output)
#         return logits

class BertForQNHackathon(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config):
        super(BertForQNHackathon, self).__init__(config)
        self.num_labels = config.num_labels
        self.robeta = RobertaModel(config)
        self.qa_outputs = nn.Linear(4*config.hidden_size, 2048)
    #    self.drop1 = nn.Dropout(0.2)
        self.qb_outputs = nn.Linear(2048, 1024)
    #    self.drop2 = nn.Dropout(0.2)
        self.qc_outputs = nn.Linear(1024, self.num_labels)
    #    self.drop3 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True, num_layers=3)
        self.linear = nn.Linear(256*2, self.num_labels)
        # self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):

        outputs = self.robeta(input_ids,
                                attention_mask=attention_mask,
    #                            token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask)
        # cls_output = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
        lstm_output, (h,c) = self.lstm(outputs[0])
        hidden = torch.cat((lstm_output[:,-1, :256], lstm_output[:,0, 256:]),dim=-1)
        #    cls_output = self.drop1(cls_output)
        # cls_output = self.qa_outputs(cls_output)
        #    cls_output = self.drop2(cls_output)
        # cls_output = self.qb_outputs(cls_output)
        #    cls_output = self.drop3(cls_output)
        # logits = self.qc_outputs(cls_output)
        linear_output = self.linear(hidden.view(-1,256*2))        
        return linear_output

if __name__ == '__main__':
    config = RobertaConfig.from_pretrained(
            'vinai/phobert-base',
            output_hidden_states=True,
            num_labels=1
            )
    mymodel = BertForQNHackathon.from_pretrained("vinai/phobert-base", config=config)
    input_ids = torch.ones((1,5), dtype=torch.int64)
    outputs = mymodel(input_ids)
    print(outputs)