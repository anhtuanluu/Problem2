import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pandas as pd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import copy
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup,AutoModelForSequenceClassification
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, TensorDataset, DataLoader
import re
import os
import warnings
from transformers import *
import time
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
tqdm.pandas()
# Todo
# add classweight
# label smoothing
# add Scheduler

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

data = pd.read_csv('data_full_hackathon.csv')
data_train = data[:16362]
data_val = data[16362:].reset_index(drop=True)
# data_train = data[:10]
# data_val = data[:10].reset_index(drop=True)
class_weight = torch.tensor([0.6175, 2.6271])

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

class CustomDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True,tokenizer=tokenizer):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = tokenizer

        self.maxlen = maxlen
        self.with_labels = with_labels 
        self.dict_label = {'relate':1,'unrelate':0}
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent = str(self.data.loc[index, 'Text'])
        sent1 = str(self.data.loc[index, 'Text1'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent1, sent, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, 'Label1']
            label = self.dict_label[label]
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids

# Creating instances of training and validation set
print("Reading training data...")
train_set = CustomDataset(data=data_train, maxlen=258, tokenizer=tokenizer)
print("Reading validation data...")
val_set = CustomDataset(data=data_val, maxlen=258, tokenizer=tokenizer)
# Creating instances of training and validation dataloaders
train_loader = DataLoader(train_set, batch_size=16)
val_loader = DataLoader(val_set, batch_size=16)

class Model_Detection(nn.Module):
    def __init__(self):
        super(Model_Detection, self).__init__()
        self.roberta = AutoModel.from_pretrained("vinai/phobert-base")
        self.classifier = torch.nn.Linear(768, 2)
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self, pair_token_ids, token_type_ids, attention_mask):
        output_1 = self.roberta(input_ids=pair_token_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model_Detection().to(device=device)

EPOCHS = 10
param_optimizer = list(model.named_parameters())
optimizer = AdamW(model.parameters(),
                      lr=2e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

num_train_optimization_steps = int(EPOCHS*len(train_loader))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_optimization_steps)
criterion = nn.CrossEntropyLoss()

train_loss = []
train_acc = []
train_f1 = []
val_loss = []
val_acc = []
val_f1 = []
best_score = 0
for epoch in range(EPOCHS):
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, EPOCHS))
    print('Training...')
    start = time.time()
    model.train()
    total_train_loss = 0
    total_train_acc  = 0
    total_train_f1 = 0
    f1_full_train = [0,0]
    for step, (pair_token, mask, seg, y) in tqdm(enumerate(train_loader)): 
        pair_token_ids = pair_token.to(device)
        mask_ids = mask.to(device)
        seg_ids = seg.to(device)
        labels = y.to(device).long()
        optimizer.zero_grad()
        # prediction = model(pair_token_ids, mask_ids, seg_ids)
        prediction = model(pair_token_ids, 
                                token_type_ids=seg_ids, 
                                attention_mask=mask_ids
                                )
        loss = criterion(prediction,labels)
        y_pred = torch.log_softmax(prediction, dim=1).argmax(dim=1).cpu().tolist()
        y_true = labels.cpu().tolist()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        f1_full_train = f1_full_train + f1_score(y_true, y_pred, average=None)


        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # scheduler.step()
        total_train_loss += loss.item()
        total_train_acc  += acc
        total_train_f1   += f1
    
    print(" Average training loss: {0:.4f}".format(total_train_loss/len(train_loader)))
    print(" Accuracy: {0:.4f}".format(total_train_acc/len(train_loader)))
    print(" F1 score: {0:.4f}".format(total_train_f1/len(train_loader)))
    print(" F1 full: ",f1_full_train/len(train_loader))
    
    train_loss.append(total_train_loss/len(train_loader))
    train_acc.append(total_train_acc/len(train_loader))
    train_f1.append(total_train_f1/len(train_loader))

    print("Running Validation...")
    model.eval()
    total_val_acc  = 0
    total_val_loss = 0
    total_val_f1 = 0
    f1_full_val = [0,0]
    y_preds = None
    y_trues = None
    for step, (pair_token, mask, seg, y) in tqdm(enumerate(val_loader)): 
        pair_token_ids = pair_token.to(device)
        mask_ids = mask.to(device)
        seg_ids = seg.to(device)
        labels = y.to(device).long()
        with torch.no_grad():
            # prediction = model(pair_token_ids, mask_ids, seg_ids)
            prediction = model(pair_token_ids, 
                                token_type_ids=seg_ids, 
                                attention_mask=mask_ids
                                )
            
            loss = criterion(prediction,labels)
            y_pred = torch.log_softmax(prediction, dim=1).argmax(dim=1).cpu().tolist()
            y_true = labels.cpu().tolist()

            y_preds = np.atleast_1d(y_pred) if y_preds is None else np.concatenate([y_preds, np.atleast_1d(y_pred)])
            y_trues = np.atleast_1d(y_true) if y_trues is None else np.concatenate([y_trues, np.atleast_1d(y_true)])

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            f1_full_val = f1_full_val + f1_score(y_true, y_pred, average=None)
            total_val_loss += loss.item()
            total_val_acc  += acc
            total_val_f1   += f1

    score = f1_score(y_trues, y_preds)
    print(" Average validation loss: {0:.4f}".format(total_val_loss/len(val_loader)))
    print(" Accuracy: {0:.4f}".format(total_val_acc/len(val_loader)))
    print(" F1 score: {0:.4f}".format(total_val_f1/len(val_loader)))
    print(" F1 full: ",f1_full_val/len(val_loader))
    print("Score: {0:.4f}, a0: {1:.4f}, a1: {2:.4f}, a2: {3:.4f}, a3: {4:.4f}, a4: {5:.4f}, a5: {6:.4f}".format(score, 
                                                                                                                f1_score(y_trues.reshape((-1, 6))[:, 0], y_preds.reshape((-1, 6))[:, 0]),
                                                                                                                f1_score(y_trues.reshape((-1, 6))[:, 1], y_preds.reshape((-1, 6))[:, 1]),
                                                                                                                f1_score(y_trues.reshape((-1, 6))[:, 2], y_preds.reshape((-1, 6))[:, 2]),
                                                                                                                f1_score(y_trues.reshape((-1, 6))[:, 3], y_preds.reshape((-1, 6))[:, 3]),
                                                                                                                f1_score(y_trues.reshape((-1, 6))[:, 4], y_preds.reshape((-1, 6))[:, 4]),
                                                                                                                f1_score(y_trues.reshape((-1, 6))[:, 5], y_preds.reshape((-1, 6))[:, 5])))
    val_loss.append(total_val_loss/len(val_loader))
    val_acc.append(total_val_acc/len(val_loader))
    val_f1.append(total_val_f1/len(val_loader))
    
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    if score >= best_score:
            torch.save(model.state_dict(),"model.bin")
            best_score = score
            np.save("y_trues.npy", y_trues)
            np.save("y_preds.npy", y_preds)