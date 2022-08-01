from models import *
from tqdm import tqdm
tqdm.pandas()
from torch import nn
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from transformers import *
import torch
import torch.utils.data
from utils import *
from config import args
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = args()

config = RobertaConfig.from_pretrained(
            args.pretrained_model_path,
            output_hidden_states=True,
            num_labels=1
            )

mymodel = BertForQNHackathon.from_pretrained(args.pretrained_model_path, config=config)
checkpoint = torch.load(os.path.join(args.checkpoint, f"model.bin"), map_location=device)
load_state_dict(mymodel, checkpoint)

if torch.cuda.is_available():
    mymodel.cuda()
    if torch.cuda.device_count():
        mymodel = nn.DataParallel(mymodel)
mymodel.eval()
data_npy = np.load(args.data_npy)
target_npy = np.load(args.target_npy)
x_train, x_test, y_train, y_test = data_npy[:2965], target_npy[:2965], data_npy[:10], target_npy[:10]

x_test = x_test[:, 0]
y_test = y_test[:, 0]

valid_dataset = torch.utils.data.TensorDataset(torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
val_preds = None
pbar = tqdm(enumerate(valid_loader),total=len(valid_loader),leave=False)
for i,(x_batch, y_batch) in pbar:
    y_pred = mymodel(x_batch.to(device), attention_mask=(x_batch > 0).to(device))
    y_pred = y_pred.squeeze().detach().cpu().numpy()
    val_preds = np.atleast_1d(y_pred) if val_preds is None else np.concatenate([val_preds, np.atleast_1d(y_pred)])
val_preds = sigmoid(val_preds)

print(y_test)
print(val_preds)