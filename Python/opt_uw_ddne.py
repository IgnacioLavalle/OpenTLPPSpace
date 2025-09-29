# Demonstration of DDNE
import time
import torch
import torch.optim as optim
from DDNE.modules import *
from DDNE.loss import *
from utils import *
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import warnings
import optuna
import pandas as pd



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of STSGN")
    #adding arguments and their respective default value
    parser.add_argument("--trials", type=int, default=300, help="Number of trials")

    return parser.parse_args()

def append_f1_with(c1f1_list, true_labels, pred_labels):
    _, _, f1_per_class, _ = \
        precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0,1], zero_division=0)
    c1f1_list.append(f1_per_class[1])



def objective(trial):
    warnings.filterwarnings("ignore")
    args = parse_args()

    lr_val = trial.suggest_categorical("lr", [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01])
    weight_decay_val = trial.suggest_categorical("weight_decay", [0.00005, 0.0001, 0.0005, 0.001])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step = 0.1) # Dropout rate
    alpha = trial.suggest_categorical("alpha", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    beta = trial.suggest_float("beta", 0.0, 1.0, step = 0.2)
    hid_dim = trial.suggest_categorical("hid_dim", [256,512,1024,2048])
    win_size = trial.suggest_categorical("win_size", [2,4,6]) # Window size of historical snapshots

    # ====================
    data_name = 'SMP22to95unweighted'
    num_nodes = 1355 # Number of nodes (Level-1 w/ fixed node set)
    num_snaps = 28 # Number of snapshots
    enc_dims = [num_nodes, hid_dim] # Layer configuration of encoder
    t_dim = hid_dim*2
    dec_dims = [2*enc_dims[-1]*win_size, t_dim, num_nodes] # Layer configuration of decoder

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)

    # ====================
    batch_size = 1 # Batch size
    num_epochs = 300 # Number of training epochs
    num_val_snaps = 3 # Number of validation snapshots
    num_test_snaps = 3 # Number of test snapshots
    num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots
    
    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True
    node_labels = np.zeros((num_nodes, 2), dtype=np.float32)
    node_labels[:137, 1] = 1.0
    node_labels[137:, 0] = 1.0 
    node_labels_tnr = torch.FloatTensor(node_labels).to(device)

    # ====================
    # Define the model
    model = DDNE(enc_dims, dec_dims, dropout_rate).to(device)
    # ==========
    # Define the optimizer
    opt = optim.Adam(model.parameters(), lr=lr_val, weight_decay=weight_decay_val)
    
    # ====================
    for epoch in range(num_epochs):
        # ====================
        # Train the model
        model.train()
        num_batch = int(np.ceil(num_train_snaps/batch_size)) # Number of batch
        total_loss = 0.0
        for b in range(num_batch):
            start_idx = b*batch_size
            end_idx = (b+1)*batch_size
            if end_idx>num_train_snaps:
                end_idx = num_train_snaps
            # ====================
            # Training for current batch
            batch_loss = 0.0
            for tau in range(start_idx, end_idx):
                # ==========
                adj_list = [] # List of historical adjacency matrices
                neigh_tnr = torch.zeros((num_nodes, num_nodes)).to(device)
                for t in range(tau-win_size, tau):
                    edges = edge_seq[t]
                    adj = get_adj_un(edges, num_nodes)
                    adj_norm = adj # Normalize the edge weights to [0, 1]
                    adj_tnr = torch.FloatTensor(adj_norm).to(device)
                    adj_list.append(adj_tnr)
                    neigh_tnr += adj_tnr
                # ==========
                edges = edge_seq[tau]
                gnd = get_adj_un(edges, num_nodes) # Training ground-truth
                gnd_norm = gnd  # Normalize the edge weights (in ground-truth) to [0, 1]
                gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
                # ==========
                adj_est, dyn_emb = model(adj_list)
                dyn_emb = torch.cat([dyn_emb, node_labels_tnr], dim=1)
                loss_ = get_DDNE_loss(adj_est, gnd_tnr, neigh_tnr, dyn_emb, alpha, beta)
                batch_loss = batch_loss + loss_
            # ==========
            # ===========================
            adj_est = adj_est.cpu().data.numpy() if torch.cuda.is_available() else adj_est.data.numpy()
            
            # Update model parameter according to batch loss
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss = total_loss + batch_loss
            
        print('Epoch %d Total Loss %f' % (epoch, total_loss))


        # ====================
        # Validate the model
        model.eval()

        c1f1_list = []
        for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
            # ====================
            adj_list = [] # List of historical adjacency matrices
            for t in range(tau-win_size, tau):
                # ==========
                edges = edge_seq[t]
                adj = get_adj_un(edges, num_nodes)
                adj_norm = adj # Normalize the edge weights to [0, 1]
                adj_tnr = torch.FloatTensor(adj_norm).to(device)
                adj_list.append(adj_tnr)
            # ====================
            # Get the prediction result
            adj_est, _ = model(adj_list)
            if torch.cuda.is_available():
                adj_est = adj_est.cpu().data.numpy()
            else:
                adj_est = adj_est.data.numpy()
            # ====================
            # Get ground-truth
            edges = edge_seq[tau]
            gnd = get_adj_un(edges, num_nodes)
            true_vals = gnd[valid_mask]
            pred_vals = adj_est[valid_mask]
            # ====================
            # Evaluate the quality of current prediction operation

            true_labels = (true_vals >= 1).astype(int)
            pred_labels = (pred_vals >= 1).astype(int)
            append_f1_with(c1f1_list, true_labels, pred_labels)
        
        val_c1_f1_mean = np.mean(c1f1_list)
        
    return val_c1_f1_mean

if __name__ == "__main__":
    args = parse_args()
    num_trials = args.trials
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)
    df = study.trials_dataframe()
    df_complete = df[df['state'] == 'COMPLETE']
    columns_of_interest = ["number", "value", "params_dropout_rate", "params_lr", "params_hid_dim", "params_alpha", "params_beta", "params_win_size"]
    df_filtered = df_complete[columns_of_interest]
    df_filtered.to_csv("optuna_trials_uw_ddne.csv", index=False)
    print("Se guardaron los trials")
