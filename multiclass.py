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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed_everything(42)
args = args()

config = AutoConfig.from_pretrained(
            args.pretrained_model_path,
            output_hidden_states=True,
            # output_attentions=True,
            num_labels=2
            )

mymodel = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model_path,config=config)
mymodel.to(device)
# mymodel = BertForQNHackathon1.from_pretrained(args.pretrained_model_path,config=config)
# mymodel.to(device)
data_npy = np.load(args.data_npy)
target_npy = np.load(args.target_npy)

target_npy =  target_npy[:, 0]

# num_train_optimization_steps = int(args.epochs*len(data_npy) / args.batch_size / args.accumulation_steps)
# optimizer = AdamW(mymodel.parameters(), lr=args.lr, correct_bias=False)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_optimization_steps)

if not os.path.exists(args.checkpoint):
    os.mkdir(args.checkpoint)

tq = tqdm(range(args.epochs + 1))
# splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(data_npy, target_npy))
splits=[0,1,2,3]
for fold, i in enumerate(splits):
    print("Training for fold {}".format(fold))
    best_score = 0
    if fold != args.fold:
        continue
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(data_npy[:2953],dtype=torch.long), torch.tensor(target_npy[:2953],dtype=torch.long))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(data_npy[2953:],dtype=torch.long), torch.tensor(target_npy[2953:],dtype=torch.long))
    tq = tqdm(range(args.epochs + 1))
    num_train_optimization_steps = int(args.epochs * len(train_dataset))
    optimizer = AdamW(mymodel.parameters(), lr=args.lr, eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=num_train_optimization_steps)
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
            mymodel.zero_grad()
            loss, logits, attentions = mymodel(x_batch.cuda(), attention_mask=(x_batch>1).cuda(),  labels=y_batch.cuda(), return_dict=False)
            lossf = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss = lossf)
            avg_loss += lossf / len(train_loader)
            # logits = logits.detach().cpu().numpy()
            # train_preds = np.atleast_1d(np.argmax(logits, axis=1)) if train_preds is None else np.concatenate([train_preds, np.atleast_1d(np.argmax(logits, axis=1))])

        mymodel.eval()
        pbar = tqdm(enumerate(valid_loader),total=len(valid_loader),leave=False)
        for i,(x_batch, y_batch) in pbar:
            with torch.no_grad():
            # y_pred = mymodel(x_batch.cuda(), attention_mask=(x_batch>1).cuda())
                loss, logits, attentions = mymodel(x_batch.cuda(), attention_mask=(x_batch>1).cuda(),  labels=y_batch.cuda(), return_dict=False)
            logits = logits.detach().cpu().numpy()
            label_ids = y_batch.to('cpu').numpy()
            avg_accuracy += flat_accuracy(logits, label_ids)
            f1 = f1_score(label_ids, np.argmax(logits, axis=1), average='macro')
            avg_eval_loss += lossf / len(valid_loader)
            val_preds = np.atleast_1d(np.argmax(logits, axis=1)) if val_preds is None else np.concatenate([val_preds, np.atleast_1d(np.argmax(logits, axis=1))])
        # val_preds = sigmoid(val_preds)
        avg_val_accuracy = avg_accuracy / len(valid_loader)
        best_th = 0
        score = f1_score(target_npy[2953:], val_preds)
        # score_train = f1_score(target_npy[:2955], train_preds)
        print(f"\nACC = {avg_val_accuracy:.4f},train F1 score @0.5 = {score:.4f}, F1 score @0.5 = {score:.4f} , Train Loss = {avg_loss:.4f}, Valid Loss = {avg_eval_loss:.4f}")
        # if score >= best_score:
        #     torch.save(mymodel.state_dict(),os.path.join(args.ckpt_path, f"model_{fold}.bin"))
        #     best_score = score