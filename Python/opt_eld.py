import argparse
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from E_LSTM_D.modules import *
from E_LSTM_D.loss import *
from utils import *
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of E-LSTM-D")
    parser.add_argument("--trials", type=int, default=150, help="Number of trials")
    return parser.parse_args()


def append_classification_metrics_with(c0precision_list, c0recall_list, c0f1_list, c1precision_list, c1recall_list, c1f1_list, true_labels, pred_labels):
    precision_per_class, recall_per_class, f1_per_class, _ = \
        precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0,1], zero_division=0)
    c0precision_list.append(precision_per_class[0])
    c1precision_list.append(precision_per_class[1])
    c0recall_list.append(recall_per_class[0])
    c1recall_list.append(recall_per_class[1])
    c0f1_list.append(f1_per_class[0])
    c1f1_list.append(f1_per_class[1])


def append_f1_with(c1f1_list, true_labels, pred_labels):
    _, _, f1_per_class, _ = \
        precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0,1], zero_division=0)
    c1f1_list.append(f1_per_class[1])


def objective(trial):
    # ====================
    # Hyperparameter search space
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step = 0.1)
    lr_val = trial.suggest_categorical("lr", [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005])
    wd_val = trial.suggest_categorical("weight_decay", [0.00001, 0.00005, 0.0001, 0.0005, 0.001])
    hid_dim = trial.suggest_categorical("hid_dim", [16,32,64,128,256])
    beta = trial.suggest_categorical("beta", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0 ])
    win_size = trial.suggest_categorical("win_size", [2,5,7,10])
    
    # ====================
    # Fixed parameters
    data_name = 'SMP22to95'
    num_nodes = 1355
    num_snaps = 28
    max_thres = 2.0
    batch_size = 1
    num_val_snaps = 3
    num_test_snaps = 3
    num_train_snaps = num_snaps - num_val_snaps - num_test_snaps
    num_epochs = 200

    # ====================
    struc_dims = [num_nodes, hid_dim*2]
    temp_dims = [struc_dims[-1], hid_dim, hid_dim]
    dec_dims = [temp_dims[-1], struc_dims[-1], num_nodes]

    edge_seq = np.load(f'data/{data_name}_edge_seq.npy', allow_pickle=True)
    valid_mask = np.zeros((num_nodes, num_nodes), dtype=bool)
    valid_mask[0:137, 137:num_nodes] = True

    # ====================
    # Model and optimizer
    model = E_LSTM_D(struc_dims, temp_dims, dec_dims, dropout_rate).to(device)
    opt = optim.Adam(model.parameters(), lr=lr_val, weight_decay=wd_val)

    best_val_f1 = -1.0

    # ====================
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
                loss_ = get_E_LSTM_D_loss(adj_est, gnd_tnr, beta)
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
    columns_of_interest = ["number", "value", "params_dropout_rate", "params_lr", "params_hid_dim"]
    df_filtered = df_complete[columns_of_interest]
    df_filtered.to_csv("optuna_trials_eld.csv", index=False)
    print("Se guardaron los trials")

if __name__ == "__main__":
    main()