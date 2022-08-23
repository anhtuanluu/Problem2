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

class BertForQNHackathon1(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config):
        super(BertForQNHackathon1, self).__init__(config)
        self.num_labels = config.num_labels
        self.robeta = AutoModel.from_pretrained("vinai/phobert-base", config=config)
        self.qa_outputs = nn.Linear(4 * config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                        start_positions=None, end_positions=None):
        outputs = self.robeta(input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            head_mask=head_mask)
        # print(outputs[2][-1].shape)
        # outputs = outputs.hidden_states
        cls_output = torch.cat((outputs[2][-1][:,0, ...],
                                outputs[2][-2][:,0, ...],
                                outputs[2][-3][:,0, ...],
                                outputs[2][-4][:,0, ...]), 
                                -1)
        logits = self.qa_outputs(cls_output)
        return logits

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


class AttentionHead(nn.Module):
    def __init__(self, h_size, hidden_dim=512):
        super().__init__()
        self.W = nn.Linear(h_size, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector

class CLRPModel(nn.Module):
    def __init__(self,transformer, config):
        super(CLRPModel,self).__init__()
        self.h_size = config.hidden_size
        self.transformer = transformer
        self.head = AttentionHead(self.h_size*4)
        self.linear = nn.Linear(self.h_size*2, 1)
        self.linear_out = nn.Linear(self.h_size*8, config.num_labels)

              
    def forward(self, input_ids, attention_mask):
        transformer_out = self.transformer(input_ids, attention_mask)
       
        all_hidden_states = torch.stack(transformer_out.hidden_states)
        cat_over_last_layers = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]),-1
        )
        
        cls_pooling = cat_over_last_layers[:, 0]   
        head_logits = self.head(cat_over_last_layers)
        y_hat = self.linear_out(torch.cat([head_logits, cls_pooling], -1))
        
        return y_hat

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        self.encoder = nn.LSTM(embedding_dim, 
                               hidden_dim, 
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               dropout=dropout)
        
        self.predictor = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, text):
        """
        The forward method is called when data is fed into the model.

        text - [tweet length, batch size]
        text_lengths - lengths of tweet
        """
        embedded = self.dropout(self.embedding(text))    

        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.encoder(embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        return self.predictor(hidden)