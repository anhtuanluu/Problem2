import torch
from torch import nn
from transformers import *

class BertForQNHackathon(nn.Module):
    # config_class = RobertaConfig
    # base_model_prefix = "roberta"

    def __init__(self, config):
        super(BertForQNHackathon, self).__init__()
        self.num_labels = config.num_labels
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base", config=config)
        self.qa_outputs = nn.Linear(2 * config.hidden_size, self.num_labels)
        # self.init_weights()

    def forward(self, input_ids, attention_mask=None, position_ids=None, head_mask=None):
        outputs = self.phobert(input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            head_mask=head_mask)
        outputs = outputs.hidden_states
        # cls_output = outputs.last_hidden_state[:,0, ...]
        cls_output = torch.cat((outputs[-1][:,0, ...],
                                outputs[-2][:,0, ...]), 
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