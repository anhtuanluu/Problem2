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
seed_everything(69)
args = args()

config = RobertaConfig.from_pretrained(
            args.pretrained_model_path,
            output_hidden_states=True,
            num_labels=1
            )
mymodel = BertForQNHackathon(config=config)
mymodel.to(device)
# mymodel = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model_path, num_labels=2)
# mymodel.to(device)

if torch.cuda.device_count():
    print(f"Training using {torch.cuda.device_count()} gpus")
    mymodel = nn.DataParallel(mymodel)
    tsfm = mymodel.module.phobert
else:
    tsfm = mymodel.phobert

data_npy = np.load(args.data_npy)
target_npy = np.load(args.target_npy)

# x_train, y_train, x_test, y_test = data_npy[:20], target_npy[:20], data_npy[:10], target_npy[:10]
x_train, y_train, x_test, y_test = data_npy[:2965], target_npy[:2965], data_npy[2965:], target_npy[2965:]

# train for a0
# target_npy =  target_npy[:, 0]

y_train = y_train[:, 2]
y_test = y_test[:, 2]
# splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(data_npy, target_npy))
# for (train_idx, test_idx) in splits:
#     x_train, y_train, x_test, y_test = data_npy[train_idx], target_npy[train_idx], data_npy[test_idx], target_npy[test_idx]
#     break

param_optimizer = list(mymodel.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

num_train_optimization_steps = int(args.epochs*len(x_train) / args.batch_size / args.accumulation_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps)
scheduler0 = get_constant_schedule(optimizer)

if not os.path.exists(args.checkpoint):
    os.mkdir(args.checkpoint)

train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
valid_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))

tq = tqdm(range(args.epochs + 1))

for child in tsfm.children():
    for param in child.parameters():
        param.requires_grad = False

frozen = True
for epoch in tq:
    if epoch > 0 and frozen:
        for child in tsfm.children():
            for param in child.parameters():
                param.requires_grad = True
        frozen = False
        del scheduler0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    val_preds = None
    best_score = 0

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    avg_loss = 0.
    avg_accuracy = 0.

    optimizer.zero_grad()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i,(x_batch, y_batch) in pbar:
        mymodel.train()
        y_pred = mymodel(x_batch.to(device), attention_mask=(x_batch > 0).to(device))
        loss =  F.binary_cross_entropy_with_logits(y_pred.view(-1).to(device), y_batch.float().to(device))
        loss = loss.mean()
        loss.backward()
        if i % args.accumulation_steps == 0 or i == len(pbar) - 1:
            optimizer.step()
            optimizer.zero_grad()
            if not frozen:
                scheduler.step()
            else:
                scheduler0.step()
        lossf = loss.item()
        pbar.set_postfix(loss = lossf)
        avg_loss += loss.item() / len(train_loader)
    print(f"\nAvg loss: {avg_loss:.4f}")

    mymodel.eval()
    pbar = tqdm(enumerate(valid_loader),total=len(valid_loader),leave=False)
    for i,(x_batch, y_batch) in pbar:
        y_pred = mymodel(x_batch.to(device), attention_mask=(x_batch > 0).to(device))
        y_pred = y_pred.squeeze().detach().cpu().numpy()
        val_preds = np.atleast_1d(y_pred) if val_preds is None else np.concatenate([val_preds, np.atleast_1d(y_pred)])
    val_preds = sigmoid(val_preds)
    best_th = 0
    score = f1_score(y_test, val_preds > 0.5)
    print(f"\nAUC = {roc_auc_score(y_test, val_preds):.4f}, F1 score = {score:.4f}")
    if score >= best_score:
        torch.save(mymodel.state_dict(),os.path.join(args.checkpoint, f"model.bin"))
        best_score = score