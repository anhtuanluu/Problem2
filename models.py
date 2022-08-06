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

class BertForQNHackathon(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config):
        super(BertForQNHackathon, self).__init__(config)
        self.num_labels = config.num_labels
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base", config=config)
        self.qa_outputs = nn.Linear(4 * config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, position_ids=None, head_mask=None):
        outputs = self.phobert(input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            head_mask=head_mask)
        outputs = outputs.hidden_states
        cls_output = torch.cat((outputs[-1][:,0, ...],
                                outputs[-2][:,0, ...],
                                outputs[-3][:,0, ...],
                                outputs[-4][:,0, ...]), 
                                -1)
        logits = self.qa_outputs(cls_output)
        return logits

# if __name__ == '__main__':
#     config = RobertaConfig.from_pretrained(
#             'vinai/phobert-base',
#             output_hidden_states=True,
#             num_labels=1
#             )
#     mymodel = BertForQNHackathon.from_pretrained("vinai/phobert-base", config=config)
#     input_ids = torch.ones((1,5), dtype=torch.int64)
#     outputs = mymodel(input_ids)