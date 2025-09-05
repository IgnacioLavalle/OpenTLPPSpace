# Demonstration of TMF

import argparse
import time
from sklearn.metrics import precision_recall_fscore_support
from TMF.TMF import *
from utils import *

import numpy as np
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of tmf")
    #adding arguments and their respective default value
    parser.add_argument("--trials", type=int, default=200, help="Number of trials")

    return parser.parse_args()

def append_f1_with(c1f1_list, true_labels, pred_labels):
    _, _, f1_per_class, _ = \
        precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0,1], zero_division=0)
    c1f1_list.append(f1_per_class[1])


def objective(trial):
    # ====================
    data_name = 'SMP22to95'
    num_nodes = 1355 # Number of nodes (Level-1 w/ fixed node set)
    num_snaps = 28 # Number of snapshots
    max_thres = 1.5 # Threshold for maximum edge weight
    hid_dim = trial.suggest_categorical("hid_dim", [128,256,512,1024]) # Dimensionality of latent space
    theta = trial.suggest_categorical("theta", [0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    alpha = trial.suggest_categorical("alpha", [0.005, 0.01, 0.05])
    beta = trial.suggest_categorical("beta", [0.001, 0.01, 0.05, 0.1])
    learn_rate = trial.suggest_categorical("lr", [0.01, 0.05, 0.1, 0.2, 0.3])
    win_size = trial.suggest_categorical("win_size", [2,4,6,8,10]) # Window size of historical snapshots

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True
    # ====================
    
    num_epochs = 500 # Number of training epochs


    # ====================
    c1f1_list = []

    for tau in range(win_size, num_snaps):
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ==========
        adj_list = [] # List of historical adjacency matrices
        for t in range(tau-win_size, tau):
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj = adj/max_thres
            adj_tnr = torch.FloatTensor(adj).to(device)
            adj_list.append(adj_tnr)
        TMF_model = TMF(num_nodes, hid_dim, win_size, num_epochs, alpha, beta, theta, learn_rate, device)
        adj_est = TMF_model.TMF_fun(adj_list)
        adj_est = adj_est*max_thres
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()

        # ==========
        # Refine prediction result
        adj_est = (adj_est+adj_est.T)/2
        # ==========
        # Evaluate the quality of current prediction operation

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
    columns_of_interest = ["number", "value", "params_lr", "params_hid_dim", "params_lambd", "params_beta", "params_alpha", "params_theta", "params_win_size"]
    df_filtered = df_complete[columns_of_interest]
    df_filtered.to_csv("optuna_trials_tmf.csv", index=False)
    print("Se guardaron los trials")