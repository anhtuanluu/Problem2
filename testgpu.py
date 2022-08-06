import torch
print(torch.cuda.is_available())
import numpy as np
from utils import *
from config import args
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed_everything(69)
args = args()

data_npy = np.load(args.data_npy)
target_npy = np.load(args.target_npy)
x_train, y_train, x_test, y_test = data_npy[:2965], target_npy[:2965], data_npy[2965:], target_npy[2965:]
y_train = y_train[:, 0]
y_test = y_test[:, 0]
print(y_train.shape)
print(y_train.sum())