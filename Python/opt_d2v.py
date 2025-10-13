# Demonstration of dyngraph2vec

import argparse
import time
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.optim as optim
from dyngraph2vec.modules import *
from dyngraph2vec.loss import *
from utils import *
import optuna


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="optuna for d2v unweighted")
    parser.add_argument("--trials", type=int, default=250, help="Number of trials")
    return parser.parse_args()

def mean_and_std_from_classlists(c0_list, c1_list):
    c0_mean = np.mean(c0_list)
    c0_std = np.std(c0_list, ddof=1) if len(c0_list) > 1 else 0.0
    c1_mean = np.mean(c1_list)
    c1_std = np.std(c1_list, ddof=1) if len(c1_list) > 1 else 0.0
    return c0_mean,c0_std,c1_mean,c1_std

def append_f1_with(c1f1_list, true_labels, pred_labels):
    _, _, f1_per_class, _ = \
        precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0,1], zero_division=0)
    c1f1_list.append(f1_per_class[1])


def objective(trial):
    
    # ====================
    data_name = 'SMP22to95'
    num_nodes = 1355 # Number of nodes (Level-1 w/ fixed node set)
    num_snaps = 28 # Number of snapshots
    max_thres = 2 # Threshold for maximum edge weight
    dim_1 = trial.suggest_categorical("hid_dim", [512, 1024,2048])
    dim_2 = dim_1 // 2
    
    struc_dims = [num_nodes, dim_1] # Layer configuration of structural encoder (FC)
    temp_dims = [dim_1, dim_2, dim_2] # Layer configuration of temporal encoder (RNN)
    dec_dims = [dim_2, dim_1, num_nodes] # Layer configuration of decoder (FC)
    beta = trial.suggest_categorical("beta", [0.0, 0.1, 0.2, 0.3, 0.4]) # Hyper-parameter of loss

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)

    # ====================
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3, 0.4, 0.5]) # Dropout rate
    win_size = trial.suggest_categorical("win_size", [2,4,6,8]) # Window size of historical snapshots
    batch_size = 1 # Batch size
    num_epochs = 250 # Number of training epochs
    num_val_snaps = 3 # Number of validation snapshots
    num_test_snaps = 3 # Number of test snapshots
    num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots
    lr_val = trial.suggest_categorical("lr", [0.00005, 0.0001, 0.0005, 0.001, 0.005])
    wdecay = trial.suggest_categorical("weight_decay", [0.0001, 0.0005, 0.001, 0.005])
    class_th = 1
    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True


    # ====================
    # Define the model
    model = dyngraph2vec(struc_dims, temp_dims, dec_dims, dropout_rate).to(device)
    # ==========
    # Define the optimizer
    opt = optim.Adam(model.parameters(), lr=lr_val, weight_decay=wdecay)

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
                adj_list = []  # List of historical adjacency matrices
                for t in range(tau-win_size, tau):
                    edges = edge_seq[t]
                    adj = get_adj_wei(edges, num_nodes, max_thres)
                    adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
                    adj_tnr = torch.FloatTensor(adj_norm).to(device)
                    adj_list.append(adj_tnr)
                # ==========
                edges = edge_seq[tau]
                gnd = get_adj_wei(edges, num_nodes, max_thres) # Training ground-truth
                gnd_norm = gnd/max_thres  # Normalize the edge weights (in ground-truth) to [0, 1]
                gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
                # ==========
                adj_est = model(adj_list)
                loss_ = get_d2v_loss(adj_est, gnd_tnr, beta)
                batch_loss = batch_loss + loss_
            # ==========
            # Update model parameter according to batch loss
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total_loss = total_loss + batch_loss
        print('Epoch %d Total Loss %f' % (epoch, total_loss))

        # ====================
        # Validate the model
        model.eval()
        RMSE_list = []
        MAE_list = []

        c0precision_list = []
        c0recall_list = []
        c0f1_list = []
        c1precision_list = []
        c1recall_list = []
        c1f1_list = []

        for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
            # ====================
            adj_list = [] # List of historical adjacency matrices
            for t in range(tau-win_size, tau):
                # ==========
                edges = edge_seq[t]
                adj = get_adj_wei(edges, num_nodes, max_thres)
                adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
                adj_tnr = torch.FloatTensor(adj_norm).to(device)
                adj_list.append(adj_tnr)
            # ====================
            # Get the prediction result
            adj_est = model(adj_list)
            if torch.cuda.is_available():
                adj_est = adj_est.cpu().data.numpy()
            else:
                adj_est = adj_est.data.numpy()
            adj_est *= max_thres # Rescale edge weights to the original value range
            # ==========
            # Refine the prediction result
            adj_est = (adj_est+adj_est.T)/2
            # ====================
            # Get ground-truth
            edges = edge_seq[tau]
            gnd = get_adj_wei(edges, num_nodes, max_thres)
            # ====================
            # Evaluate the quality of current prediction operation
            true_vals = gnd[valid_mask]
            pred_vals = adj_est[valid_mask]

            true_labels = (true_vals >= 1).astype(int)
            pred_labels = (pred_vals >= class_th).astype(int)

def main():
    args = parse_args()
    num_trials = args.trials
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)
    df = study.trials_dataframe()
    df_complete = df[df['state'] == 'COMPLETE']
    columns_of_interest = ["number", "value", "params_dropout_rate", "params_lr", "params_hid_dim", "params_beta", "params_win_size", "params_weight_decay"]
    df_filtered = df_complete[columns_of_interest]
    df_filtered.to_csv("optuna_trials_d2v.csv", index=False)
    print("Se guardaron los trials")


if __name__ == "__main__":
    main()
