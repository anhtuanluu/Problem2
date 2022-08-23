from models import *
from tqdm import tqdm
tqdm.pandas()
from torch import nn
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, r2_score
from transformers import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from utils import *
from config import args
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler

labels_scaler = StandardScaler()

def create_optimizer(model):
    named_parameters = list(model.named_parameters())    
    
    roberta_parameters = named_parameters[:389]    
    attention_parameters = named_parameters[391:395]
    regressor_parameters = named_parameters[395:]
        
    attention_group = [params for (name, params) in attention_parameters]
    regressor_group = [params for (name, params) in regressor_parameters]
    
    parameters = []
    parameters.append({"params": attention_group})
    parameters.append({"params": regressor_group})
    # increase lr every second layer
    increase_lr_every_k_layer = 1
    lrs = np.linspace(1, 5, 24 // increase_lr_every_k_layer)
    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01
        splitted_name = name.split('.')
        lr = 1e-5
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[3]):
            layer_num = int(splitted_name[3])
            lr = lrs[layer_num // increase_lr_every_k_layer] * 1e-5 

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    return AdamW(parameters)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed_everything(42)
args = args()

config = AutoConfig.from_pretrained(
            args.pretrained_model_path,
            output_hidden_states=True,
            # output_attentions=True,
            num_labels=6
            )

transformer = AutoModel.from_pretrained(args.pretrained_model_path, config=config)  

mymodel = CLRPModel(transformer, config)
mymodel.to(device)
data_npy = np.load(args.data_npy)
target_npy = np.load(args.target_npy)

y_new = get_new_labels(np.where(target_npy > 0, 1, 0))
criterion = nn.MSELoss()

if not os.path.exists(args.checkpoint):
    os.mkdir(args.checkpoint)

tq = tqdm(range(args.epochs + 1))
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(data_npy, y_new))
for fold, (train_dx, test_dx) in enumerate(splits):
    print("Training for fold {}".format(fold))
    best_score = 99999
    if fold != args.fold:
        continue
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(data_npy[train_dx],dtype=torch.long), torch.tensor(target_npy[train_dx],dtype=torch.long))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(data_npy[test_dx],dtype=torch.long), torch.tensor(target_npy[test_dx],dtype=torch.long))
    tq = tqdm(range(args.epochs + 1))
    num_train_optimization_steps = int(args.epochs * len(train_dataset)/args.batch_size/args.accumulation_steps)
    optimizer = create_optimizer(mymodel)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_train_optimization_steps * 0.1), num_training_steps=num_train_optimization_steps)
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
            logits = mymodel(x_batch.cuda(), attention_mask=(x_batch>1).cuda())
            loss = criterion(logits, y_batch.type_as(logits))
            lossf = loss.item()
            loss.backward()
            if i % args.accumulation_steps == 0 or i == len(pbar) - 1:
            # torch.nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            pbar.set_postfix(loss = lossf)
            avg_loss += lossf / len(train_loader)

        mymodel.eval()
        pbar = tqdm(enumerate(valid_loader),total=len(valid_loader),leave=False)
        for i,(x_batch, y_batch) in pbar:
            with torch.no_grad():
                logits = mymodel(x_batch.cuda(), attention_mask=(x_batch>1).cuda())
            # loss = criterion(logits.squeeze(), y_batch.type_as(logits))
            loss = criterion(logits, y_batch.type_as(logits))
            lossf = loss.item()
            avg_eval_loss += lossf / len(valid_loader)
            logits = logits.detach().cpu().numpy()
            val_preds = logits if val_preds is None else np.concatenate([val_preds, logits])
            pbar.set_postfix(loss = lossf)
        if avg_eval_loss <= best_score:
            torch.save(mymodel.state_dict(), f"/content/drive/MyDrive/Models/model_{fold}.bin")
            best_score = avg_eval_loss
        print(f"\nTrain Loss = {avg_loss:.4f}, Valid Loss = {avg_eval_loss:.4f}, Best = {best_score:.4f}")
        print(f"R2 Score = {r2_score(target_npy[test_dx], val_preds):.4f}")
        print(f"MAE = {mean_absolute_error(target_npy[test_dx], val_preds):.4f}")
        print(f"MSE = {mean_squared_error(target_npy[test_dx], val_preds):.4f}")