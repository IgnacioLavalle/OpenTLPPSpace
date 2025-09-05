# Demonstration of LIST

import argparse
import time

from sklearn.metrics import precision_recall_fscore_support
from LIST.LIST import *
from utils import *

import numpy as np
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of List")
    #adding arguments and their respective default value
    parser.add_argument("--trials", type=int, default=200, help="Number of trials")

    return parser.parse_args()

def append_f1_with(c1f1_list, true_labels, pred_labels):
    _, _, f1_per_class, _ = \
        precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0,1], zero_division=0)
    c1f1_list.append(f1_per_class[1])


def objective(trial):
    hid_dim = trial.suggest_categorical("hid_dim", [32,64,128,256,512]) # Dimensionality of latent space
    theta = trial.suggest_categorical("theta", [2.0,5.0,7.0])
    beta = trial.suggest_categorical("beta", [0.001, 0.01, 0.1])
    lambd = trial.suggest_categorical("lambd", [0.001, 0.01, 0.1, 0.2])
    learn_rate = trial.suggest_categorical("lr", [0.001, 0.005, 0.01, 0.05, 0.1])
    win_size = trial.suggest_categorical("win_size", [3,5,7,9]) # Window size of historical snapshots
    num_epochs = trial.suggest_categorical("epochs", [500,750,1000]) # Number of training epochs

    # ====================
    data_name = 'SMP22to95'
    num_nodes = 1355 # Number of nodes (Level-1 w/ fixed node set)
    num_snaps = 28 # Number of snapshots
    max_thres = 2.0 # Threshold for maximum edge weight
    b_iterations = 100

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True

    # ====================
    
    dec_list = get_dec_list(win_size, theta) # Get the list of decaying factors

    # ====================

    c1f1_list = []


    for tau in range(win_size, num_snaps):
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ==========
        adj_list = [] # List of historical adjacency matrices
        P_list = [] # List of regularization matrices
        for t in range(tau - win_size, tau):
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_tnr = torch.FloatTensor(adj).to(device)
            adj_list.append(adj_tnr)
            P = get_P(adj, num_nodes, lambd, b_iterations, device=device)
            P_list.append(P)
        LIST_model = LIST(num_nodes, hid_dim, win_size, dec_list, P_list, num_epochs, beta, learn_rate, device)
        adj_est = LIST_model.LIST_fun(adj_list)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()

        # ==========
        # Evaluate the quality of current prediction operation
                # Evaluate the prediction result
        true_vals = gnd[valid_mask]
        pred_vals = adj_est[valid_mask]

        true_labels = (true_vals >= 1).astype(int)
        pred_labels = (pred_vals >= 1).astype(int)

        append_f1_with(c1f1_list, true_labels, pred_labels)

        val_c1_f1_mean = np.mean(c1f1_list)


    return val_c1_f1_mean 

if __name__ == '__main__':
    args = parse_args()
    num_trials = args.trials
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)
    df = study.trials_dataframe()
    df_complete = df[df['state'] == 'COMPLETE']
    columns_of_interest = ["number", "value", "params_lr", "params_hid_dim", "params_lambd", "params_beta", "params_theta", "params_win_size", "params_epochs"]
    df_filtered = df_complete[columns_of_interest]
    df_filtered.to_csv("optuna_trials_list.csv", index=False)
    print("Se guardaron los trials")
