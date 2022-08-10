from models import *
from tqdm import tqdm
tqdm.pandas()
from torch import nn
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from transformers import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from utils import *
from config import args
from prettytable import PrettyTable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed_everything(42)
args = args()

def check_metric_vaild(y_pred, y_true):
    if y_true.min() == y_true.max() == 0:   # precision
        return False
    if y_pred.min() == y_pred.max() == 0:   # recall
        return False
    return True
config = RobertaConfig.from_pretrained(
            args.pretrained_model_path,
            output_hidden_states=True,
            # output_attentions=True,
            num_labels=5
            )

# mymodel = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model_path,config=config)

class BertForQNHackathon(nn.Module):
    def __init__(self, config):
        super(BertForQNHackathon, self).__init__()
        self.num_labels = config.num_labels
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base", config=config)
        self.qa_outputs = nn.Linear(4 * config.hidden_size, self.num_labels)
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.phobert(input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            head_mask=head_mask)
        cls_output = torch.cat((outputs[2][-1][:,0, ...],
                                outputs[2][-2][:,0, ...],
                                outputs[2][-3][:,0, ...],
                                outputs[2][-4][:,0, ...]), 
                                -1)
        logits = self.qa_outputs(cls_output)
        logits = torch.sigmoid(logits)
        loss = 0
        if labels is not None:
            loss = self.criterion(logits, labels)
        return loss, logits

class MultiClass(nn.Module):
  def __init__(self):
    super().__init__()
    self.bert =  AutoModelForSequenceClassification.from_pretrained(args.pretrained_model_path,config=config)
    self.criterion = nn.BCELoss()
  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    # output = self.classifier(output.logits)
    output = torch.sigmoid(output.logits)
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output

# mymodel = MultiClass()
mymodel = BertForQNHackathon(config=config)
mymodel.to(device)

data_npy = np.load(args.data_npy)
target_npy = np.load(args.target_npy)

target_npy = target_npy[:, 1]
x = target_npy > 0
target_npy =  target_npy[x > 0] - 1
data_npy = data_npy[x > 0]
target_npy_onehot = np.zeros((target_npy.size, target_npy.max()+1))
target_npy_onehot[np.arange(target_npy.size), target_npy] = 1

if not os.path.exists(args.checkpoint):
    os.mkdir(args.checkpoint)

tq = tqdm(range(args.epochs + 1))
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(data_npy, target_npy))
for fold, (i, j) in enumerate(splits):
    print("Training for fold {}".format(fold))
    best_score = 0
    if fold != args.fold:
        continue
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(data_npy[i],dtype=torch.long), torch.tensor(target_npy_onehot[i],dtype=torch.float))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(data_npy[j],dtype=torch.long), torch.tensor(target_npy_onehot[j],dtype=torch.float))
    tq = tqdm(range(args.epochs + 1))
    attribute_list = ['a1', 'a2', 'a3', 'a4', 'a5']
    num_train_optimization_steps = int(args.epochs*len(train_dataset))
    optimizer = AdamW(mymodel.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_optimization_steps)
    for epoch in tq:

        val_preds = None
        train_preds = None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        avg_loss = 0.
        avg_eval_loss = 0
        avg_accuracy = 0.
    
        optimizer.zero_grad()
        mymodel.train()

        pbar = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
        for i,(x_batch, y_batch) in pbar:
            optimizer.zero_grad()
            loss, logits = mymodel(x_batch.cuda(), attention_mask=(x_batch>1).cuda(),  labels=y_batch.cuda())
            lossf = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)

        mymodel.eval()

        preds_tensor = np.empty(shape=[0, 5], dtype=np.byte)   
        labels_tensor = np.empty(shape=[0, 5], dtype=np.byte)   

        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        average_precision = 0.0
        average_recall = 0.0
        average_f1score = 0.0
        valid_count = 0

        pbar = tqdm(enumerate(valid_loader),total=len(valid_loader),leave=False)
        for i,(x_batch, y_batch) in pbar:
            # y_pred = mymodel(x_batch.cuda(), attention_mask=(x_batch>1).cuda())
            loss, logits = mymodel(x_batch.cuda(), attention_mask=(x_batch>1).cuda(),  labels=y_batch.cuda())
            preds = torch.gt(logits, torch.ones_like(logits)/2)
            preds = preds.detach().cpu().numpy()
            labels = y_batch.to('cpu').numpy()
            preds_tensor = np.append(preds_tensor, preds, axis=0)
            labels_tensor = np.append(labels_tensor, labels, axis=0)
        
        for i, name in enumerate(attribute_list):
            y_true, y_pred = labels_tensor[:, i], preds_tensor[:, i]
            accuracy_list.append(accuracy_score(y_true, y_pred))
            if check_metric_vaild(y_pred, y_true):    # exclude ill-defined cases
                precision_list.append(precision_score(y_true, y_pred, average='binary'))
                recall_list.append(recall_score(y_true, y_pred, average='binary'))
                f1_score_list.append(f1_score(y_true, y_pred, average='binary'))
                average_precision += precision_list[-1]
                average_recall += recall_list[-1]
                average_f1score += f1_score_list[-1]
                valid_count += 1
            else:
                precision_list.append(-1)
                recall_list.append(-1)
                f1_score_list.append(-1)
        if valid_count == 0:
            valid_count = 1
        average_acc = np.mean(accuracy_list)
        average_precision = average_precision / valid_count
        average_recall = average_recall / valid_count
        average_f1score = average_f1score / valid_count
        print('Average accuracy: {:.4f}'.format(average_acc))
        print('Average f1 score: {:.4f}'.format(average_f1score))
        
        table = PrettyTable(['attribute', 'accuracy', 'precision', 'recall', 'f1 score'])
        for i, name in enumerate(attribute_list):
            table.add_row([name,
                '%.3f' % accuracy_list[i],
                '%.3f' % precision_list[i] if precision_list[i] >= 0.0 else '-',
                '%.3f' % recall_list[i] if recall_list[i] >= 0.0 else '-',
                '%.3f' % f1_score_list[i] if f1_score_list[i] >= 0.0 else '-',
                ])
        print(table)