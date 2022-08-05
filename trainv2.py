from models import *
from tqdm import tqdm
tqdm.pandas()
from torch import nn
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from transformers import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from utils import *
from config import args
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed_everything(69)
args = args()

config = RobertaConfig.from_pretrained(
            args.pretrained_model_path,
            output_hidden_states=True,
            num_labels=2
            )
# model = BertForQNHackathon(config=config)
# model.to(device)
model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model_path, num_labels=2)
model.to(device)


data_npy = np.load(args.data_npy)
target_npy = np.load(args.target_npy)
# x_train, y_train, x_test, y_test = data_npy[:20], target_npy[:20], data_npy[:10], target_npy[:10]
# x_train, y_train, x_test, y_test = data_npy[:2965], target_npy[:2965], data_npy[2965:], target_npy[2965:]

# train for a0
# y_train = y_train[:, 0]
# y_test = y_test[:, 0]
target_npy =  target_npy[:, 0]

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(data_npy, target_npy))
for (train_idx, test_idx) in splits:
    x_train, y_train, x_test, y_test = data_npy[train_idx], target_npy[train_idx], data_npy[test_idx], target_npy[test_idx]
    break

criterion = nn.CrossEntropyLoss()
param_optimizer = list(model.named_parameters())
optimizer = AdamW(model.parameters(),
                      lr=3e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
valid_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

EPOCHS = 10

train_loss = []
train_acc = []
train_f1 = []
val_loss = []
val_acc = []
val_f1 = []
for epoch in range(EPOCHS):
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, EPOCHS))
    print('Training...')
    start = time.time()
    model.train()
    total_train_loss = 0
    total_train_acc  = 0
    total_train_f1 = 0
    f1_full_train = [0,0]
    for step, (pair_token, y) in tqdm(enumerate(train_loader)): 
        pair_token_ids = pair_token.to(device)
        # mask_ids = mask.to(device)
        # seg_ids = seg.to(device)
        labels = y.to(device).long()
        optimizer.zero_grad()
        prediction = model(pair_token_ids, 
                                # token_type_ids=seg_ids, 
                                attention_mask=(pair_token_ids > 1)
                                )
        loss = criterion(prediction,labels)
        y_pred = torch.log_softmax(prediction, dim=1).argmax(dim=1).cpu().tolist()
        y_true = labels.cpu().tolist()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        f1_full_train = f1_full_train + f1_score(y_true, y_pred, average=None)


        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
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
    for step, (pair_token, y) in tqdm(enumerate(valid_loader)): 
        pair_token_ids = pair_token.to(device)
        # mask_ids = mask.to(device)
        # seg_ids = seg.to(device)
        labels = y.to(device).long()
        with torch.no_grad():
            # prediction = model(pair_token_ids, mask_ids, seg_ids)
            prediction = model(pair_token_ids, 
                                # token_type_ids=seg_ids, 
                                attention_mask=(pair_token_ids > 0)
                                )
            
            loss = criterion(prediction,labels)
            y_pred = torch.log_softmax(prediction, dim=1).argmax(dim=1).cpu().tolist()
            y_true = labels.cpu().tolist()

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            f1_full_val = f1_full_val + f1_score(y_true, y_pred, average=None)
            
            total_val_loss += loss.item()
            total_val_acc  += acc
            total_val_f1   += f1

    print(" Average validation loss: {0:.4f}".format(total_val_loss/len(valid_loader)))
    print(" Accuracy: {0:.4f}".format(total_val_acc/len(valid_loader)))
    print(" F1 score: {0:.4f}".format(total_val_f1/len(valid_loader)))
    print(" F1 full: ",f1_full_val/len(valid_loader))
    val_loss.append(total_val_loss/len(valid_loader))
    val_acc.append(total_val_acc/len(valid_loader))
    val_f1.append(total_val_f1/len(valid_loader))
    
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
