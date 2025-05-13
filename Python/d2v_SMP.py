# Demonstration of dyngraph2vec

import time
import torch
import torch.optim as optim
from dyngraph2vec.modules import *
from dyngraph2vec.loss import *
from utils import *
import argparse
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report,
    precision_recall_curve, average_precision_score
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of D2V")
    #adding arguments and their respective default value

    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--epsilon", type=int, default=2, help="Threshold of zero-refining (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs (default: 100)")
    parser.add_argument("--num_val_snaps", type=int, default=3, help="Number of validation snapshots (default: 3)")
    parser.add_argument("--num_test_snaps", type=int, default=3, help="Number of test snapshots (default: 3)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay (default: 5e-4)")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta value (default: 0.1)")
    parser.add_argument("--win_size", type=int, default=2, help="Window size of historical snapshots (default: 2)")
    parser.add_argument("--max_thres", type=float, default=2.0, help="Threshold for maximum edge weight (default: 1) (el maximo del grafo es 17500)")
    parser.add_argument("--save_forecast", type=bool, default=False, help="Indicates whether you want or not to save the forecast result")
    parser.add_argument("--save_metrics", type=bool, default=True, help="Indicates whether you want or not to save the classification metrics json")


    return parser.parse_args()

# ====================
def main():
    start_time = time.time()

    args = parse_args()

    data_name = 'SMP22to95'
    num_nodes = 1355 # Number of nodes (Level-1 w/ fixed node set)
    num_snaps = 28 # Number of snapshots
    max_thres = args.max_thres # Threshold for maximum edge weight
    struc_dims = [num_nodes, 32] # Layer configuration of structural encoder (FC)
    temp_dims = [struc_dims[-1], 16, 16] # Layer configuration of temporal encoder (RNN)
    dec_dims = [temp_dims[-1], 32, num_nodes] # Layer configuration of decoder (FC)
    beta = args.beta # Hyper-parameter of loss

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)

    # ====================
    dropout_rate = args.dropout_rate # Dropout rate
    epsilon = args.epsilon # Threshold of zero-refining
    win_size = args.win_size # Window size of historical snapshots
    batch_size = args.batch_size # Batch size
    num_epochs = args.num_epochs # Number of training epochs
    num_val_snaps = args.num_val_snaps # Number of validation snapshots
    num_test_snaps = args.num_test_snaps # Number of test snapshots
    num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots
    lr_val = args.lr
    weight_decay_val = args.weight_decay

    max_country = 136 #Max country index is 136
    min_product = 137  # Minimum product index is 137
    max_product = 1354 #max product index is 1354 
    total_nodes = max_product + 1  #so we have 1355 total nodes
    
    valid_mask = np.zeros((total_nodes, total_nodes), dtype=bool)

    valid_mask[0:137, 137:1355] = True

    # ====================
    # Define the model
    model = dyngraph2vec(struc_dims, temp_dims, dec_dims, dropout_rate).to(device)
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
            for r in range(num_nodes):
                adj_est[r, r] = 0
            for r in range(num_nodes):
                for c in range(num_nodes):
                    if adj_est[r, c] <= epsilon:
                        adj_est[r, c] = 0
            # ====================
            # Get ground-truth
            edges = edge_seq[tau]
            gnd = get_adj_wei(edges, num_nodes, max_thres)
            # ====================
            # Evaluate the quality of current prediction operation
            true_vals = gnd[valid_mask]
            pred_vals = adj_est[valid_mask]

            true_labels = (true_vals >= 1).astype(int)
            pred_scores = pred_vals
            pred_labels = (pred_vals >= 1).astype(int)

            #Errors
            abs_errors = np.abs(pred_vals - true_vals)
            sq_errors = (pred_vals - true_vals) ** 2

            #MAE and std
            filtered_mae = np.mean(abs_errors)
            mae_std = np.std(abs_errors)

            #RMSE and std
            filtered_rmse = np.sqrt(np.mean(sq_errors))
            rmse_std = np.std(sq_errors)


            ###
            RMSE = get_RMSE(adj_est, gnd, num_nodes)
            MAE = get_MAE(adj_est, gnd, num_nodes)
            RMSE_list.append(RMSE)
            MAE_list.append(MAE)
        # ====================
        RMSE_mean = np.mean(RMSE_list)
        RMSE_std = np.std(RMSE_list, ddof=1)
        MAE_mean = np.mean(MAE_list)
        MAE_std = np.std(MAE_list, ddof=1)
        print('Val Epoch %d RMSE %f %f MAE %f %f' % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std))
        print(f'Val Epoch {epoch} filtered RMSE: {filtered_rmse} std: {rmse_std} filtered MAE: {filtered_mae} std: {mae_std}' )
        print(classification_report(true_labels, pred_labels))


        # ====================
        # Test the model
        model.eval()
        RMSE_list = []
        MAE_list = []
        for tau in range(num_snaps-num_test_snaps, num_snaps):
            # ====================
            adj_list = []  # List of historical adjacency matrices
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
            adj_est *= max_thres # Rescale the edge weights to the original value range
            # ==========
            # Refine the prediction result
            adj_est = (adj_est+adj_est.T)/2
            for r in range(num_nodes):
                adj_est[r, r] = 0
            for r in range(num_nodes):
                for c in range(num_nodes):
                    if adj_est[r, c] <= epsilon:
                        adj_est[r, c] = 0
            # ====================
            # Get the ground-truth
            edges = edge_seq[tau]
            gnd = get_adj_wei(edges, num_nodes, max_thres)
            # ====================
            # Evaluate the quality of current prediction operation

            true_vals = gnd[valid_mask]
            pred_vals = adj_est[valid_mask]

            true_labels = (true_vals >= 1).astype(int)
            pred_scores = pred_vals
            pred_labels = (pred_vals >= 1).astype(int)

            #Errors
            abs_errors = np.abs(pred_vals - true_vals)
            sq_errors = (pred_vals - true_vals) ** 2

            #MAE and std
            filtered_mae = np.mean(abs_errors)
            mae_std = np.std(abs_errors)

            #RMSE and std
            filtered_rmse = np.sqrt(np.mean(sq_errors))
            rmse_std = np.std(sq_errors)


            RMSE = get_RMSE(adj_est, gnd, num_nodes)
            MAE = get_MAE(adj_est, gnd, num_nodes)
            RMSE_list.append(RMSE)
            MAE_list.append(MAE)
        # ====================
        RMSE_mean = np.mean(RMSE_list)
        RMSE_std = np.std(RMSE_list, ddof=1)
        MAE_mean = np.mean(MAE_list)
        MAE_std = np.std(MAE_list, ddof=1)
        print('Test Epoch %d RMSE %f %f MAE %f %f' % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std))
        print(f'Val Epoch {epoch} filtered RMSE: {filtered_rmse} std: {rmse_std} filtered MAE: {filtered_mae} std: {mae_std}' )
        print(classification_report(true_labels, pred_labels))

        print()
    print('Total runtime was: %s seconds' % (time.time() - start_time))

if __name__ == "__main__":
    main()