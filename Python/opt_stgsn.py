# Demonstration of STGSN

import argparse
import time
import torch
import torch.optim as optim
from STGSN.modules import *
from STGSN.loss import *
from utils import *
from sklearn.metrics import precision_recall_fscore_support
import optuna
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of STSGN")
    #adding arguments and their respective default value
    parser.add_argument("--trials", type=int, default=90, help="Number of trials")

    return parser.parse_args()

def append_f1_with(c1f1_list, true_labels, pred_labels):
    _, _, f1_per_class, _ = \
        precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0,1], zero_division=0)
    c1f1_list.append(f1_per_class[1])


def objective(trial):

    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step = 0.1)
    lr = trial.suggest_categorical("lr", [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01])
    weight_decay = trial.suggest_categorical("weight_decay", [0.00005, 0.0001, 0.0005, 0.001])
    enc_dim = trial.suggest_categorical("enc_dim", [16,32,64,128])
    theta = trial.suggest_categorical("theta", [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0 ])
    win_size = trial.suggest_categorical("win_size", [2,4,6])

    # ====================
    f_layer = s_layer = t_layer = enc_dim
    #If you choose enconder dimension to be the same you have [feat_dim, f_layer,f_layer,f_layer]
    #So if instead you choose encoder dimensions to be different, you have [feat_dim, f_layer,f_layer * 2,f_layer * 4]
    s_layer *= 2
    t_layer *= 4

    data_name = 'SMP22to95'
    feat_name = 'SMP22to95_oh'
    num_nodes = 1355 # Number of nodes (Level-1 w/ fixed node set)
    num_snaps = 28 # Number of snapshots
    max_thres = 2 # Threshold for maximum edge weight
    feat_dim = 32 # Dimensionality of feature input
    enc_dims = [feat_dim, f_layer, s_layer, t_layer] # Layer configuration of encoder
    emb_dim = enc_dims[-1] # Dimensionality of dynamic embedding
    win_size = win_size # Window size of historical snapshots
    theta = theta # Hyper-parameter for collapsed graph

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
    feat = np.load('data/%s_feat.npy' % (feat_name), allow_pickle=True)
    
    feat_tnr = torch.FloatTensor(feat).to(device)

    feat_list = []
    for i in range(win_size):
        feat_list.append(feat_tnr)

    # ====================
    dropout_rate = dropout_rate # Dropout rate
    batch_size = 1 # Batch size
    num_epochs = 200 # Number of training epochs
    num_val_snaps = 3 # Number of validation snapshots
    num_test_snaps = 3 # Number of test snapshots
    num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots

    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True

    # ====================
    # Define the model
    model = STGSN(enc_dims, dropout_rate).to(device)
    # ==========
    # Define the optimizer
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ====================
    for epoch in range(num_epochs):
        # ====================
        # Pre-train the model
        model.train()
        num_batch = int(np.ceil(num_train_snaps/batch_size))  # Number of batch
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
                sup_list = []  # List of GNN support (tensor)
                col_net = np.zeros((num_nodes, num_nodes))
                coef_sum = 0.0
                for t in range(tau-win_size, tau):
                    # ==========
                    edges = edge_seq[t]
                    adj = get_adj_wei(edges, num_nodes, max_thres)
                    adj_norm = adj/max_thres
                    sup = get_gnn_sup_d(adj_norm)
                    sup_sp = sp.sparse.coo_matrix(sup)
                    sup_sp = sparse_to_tuple(sup_sp)
                    idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
                    vals = torch.FloatTensor(sup_sp[1]).to(device)
                    #sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
                    sup_tnr = torch.sparse_coo_tensor(idxs.t(), vals, torch.Size(sup_sp[2]), dtype=torch.float32, device=device)
                    sup_list.append(sup_tnr)
                    # ==========
                    coef = (1-theta)**(tau-t)
                    col_net += coef*adj_norm
                    coef_sum += coef
                # ==========
                col_net /= coef_sum
                col_sup = get_gnn_sup_d(col_net)
                col_sup_sp = sp.sparse.coo_matrix(col_sup)
                col_sup_sp = sparse_to_tuple(col_sup_sp)
                idxs = torch.LongTensor(col_sup_sp[0].astype(float)).to(device)
                vals = torch.FloatTensor(col_sup_sp[1]).to(device)
                #col_sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, col_sup_sp[2]).float().to(device)
                col_sup_tnr = torch.sparse_coo_tensor(idxs.t(), vals, torch.Size(col_sup_sp[2]), dtype=torch.float32, device=device)
                # ==========
                edges = edge_seq[tau]
                gnd = get_adj_wei(edges, num_nodes, max_thres) # Training ground-truth
                gnd_norm = gnd/max_thres # Normalize the edge weights (in ground-truth) to [0, 1]
                gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
                # ==========
                adj_est = model(sup_list, feat_list, col_sup_tnr, feat_tnr, num_nodes)
                loss_ = get_STGSN_loss_wei(adj_est, gnd_tnr)
                batch_loss = batch_loss + loss_
            # ===========
            # Update model parameter according to batch loss
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total_loss = total_loss + batch_loss
        print('Epoch %d Total Loss %f' % (epoch, total_loss))

        # ====================
        # Validate the model
        model.eval()
        # ==========
        c1f1_list = []

        for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
            # ====================
            sup_list = [] # List of GNN support (tensor)
            col_net = np.zeros((num_nodes, num_nodes))
            coef_sum = 0.0
            for t in range(tau-win_size, tau):
                # ==========
                edges = edge_seq[t]
                adj = get_adj_wei(edges, num_nodes, max_thres)
                adj_norm = adj / max_thres
                sup = get_gnn_sup_d(adj_norm)
                sup_sp = sp.sparse.coo_matrix(sup)
                sup_sp = sparse_to_tuple(sup_sp)
                idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
                vals = torch.FloatTensor(sup_sp[1]).to(device)
                #sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
                sup_tnr = torch.sparse_coo_tensor(idxs.t(), vals, torch.Size(sup_sp[2]), dtype=torch.float32, device=device)
                sup_list.append(sup_tnr)
                # ==========
                coef = (1-theta)**(tau-t)
                col_net += coef*adj_norm
                coef_sum += coef
            # ==========
            col_net /= coef_sum
            col_sup = get_gnn_sup_d(col_net)
            col_sup_sp = sp.sparse.coo_matrix(col_sup)
            col_sup_sp = sparse_to_tuple(col_sup_sp)
            idxs = torch.LongTensor(col_sup_sp[0].astype(float)).to(device)
            vals = torch.FloatTensor(col_sup_sp[1]).to(device)
            #col_sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, col_sup_sp[2]).float().to(device)
            col_sup_tnr = torch.sparse_coo_tensor(idxs.t(), vals, torch.Size(col_sup_sp[2]), dtype=torch.float32, device=device)
            # ==========
            # Get the prediction result
            adj_est = model(sup_list, feat_list, col_sup_tnr, feat_tnr, num_nodes)
            if torch.cuda.is_available():
                adj_est = adj_est.cpu().data.numpy()
            else:
                adj_est = adj_est.data.numpy()
            adj_est *= max_thres  # Rescale the edge weights to the original value range
            # ==========
            # Refine the prediction result
            adj_est = (adj_est+adj_est.T)/2

            # ====================
            # Get the ground-truth
            edges = edge_seq[tau]
            gnd = get_adj_wei(edges, num_nodes, max_thres)
            # ====================
            # ====================
            # Evaluate the prediction result
            true_vals = gnd[valid_mask]
            pred_vals = adj_est[valid_mask]

            true_labels = (true_vals >= 1).astype(int)
            pred_labels = (pred_vals >= 1).astype(int)

            append_f1_with(c1f1_list, true_labels, pred_labels)
        
        val_c1_f1_mean = np.mean(c1f1_list)
        
    return val_c1_f1_mean

def main():
    args = parse_args()
    num_trials = args.trials
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)
    df = study.trials_dataframe()
    df_complete = df[df['state'] == 'COMPLETE']
    columns_of_interest = ["number", "value", "params_dropout_rate", "params_lr", "params_enc_dim", "params_theta", "params_win_size"]
    df_filtered = df_complete[columns_of_interest]
    df_filtered.to_csv("optuna_trials_stgsn.csv", index=False)
    print("Se guardaron los trials")


if __name__ == "__main__":
    main()
