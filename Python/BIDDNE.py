# Demonstration of BIDDNE
import random
import time
import torch
import torch.optim as optim
from BIDDNE.modules import *
from BIDDNE.loss import *
from utils import *
import argparse
import numpy as np
import json
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report,
    precision_recall_curve, average_precision_score
)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of BIDDNE")
    #adding arguments and their respective default value

    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--epsilon", type=int, default=2, help="Threshold of zero-refining (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs (default: 100)")
    parser.add_argument("--num_val_snaps", type=int, default=3, help="Number of validation snapshots (default: 3)")
    parser.add_argument("--num_test_snaps", type=int, default=3, help="Number of test snapshots (default: 3)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (default: 1e-4)")
    parser.add_argument("--alpha", type=float, default=3.0, help="Alpha value (default: 2.0)")
    parser.add_argument("--beta", type=float, default=0.0, help="Alpha value (default: 0.2)")
    parser.add_argument("--win_size", type=int, default=2, help="Window size of historical snapshots (default: 2)")
    parser.add_argument("--max_thres", type=float, default=2.0, help="Threshold for maximum edge weight (default: 1) (el maximo del grafo es 17500)")
    parser.add_argument("--save_forecast", type=bool, default=False, help="Indicates whether you want or not to save the forecast result")
    parser.add_argument("--save_metrics", type=bool, default=True, help="Indicates whether you want or not to save the classification metrics json")
    parser.add_argument("--filter", type=bool, default=False, help="Indicates whether you want or not to filter edges below 0.1")
    parser.add_argument("--wc", type=float, default=5.0, help="Indicates weight of class 1")

    


    return parser.parse_args()

def get_RMSE_(pred, true, *args):
    return np.sqrt(np.mean((pred - true) ** 2))

def get_MAE_(pred, true, *args):
    return np.mean(np.abs(pred - true))


def main():
    start_time = time.time()
    args = parse_args()
    save_forecast = args.save_forecast
    save_metrics = args.save_metrics
    # ====================
    data_name = 'SMP22to95'
    num_nodes = 1355 # Number of nodes (Level-1 w/ fixed node set)
    num_snaps = 28 # Number of snapshots
    max_thres = args.max_thres # Threshold for maximum edge weight
    win_size = args.win_size # Window size of historical snapshots
    latent_embedding_dim = 128
    enc_dims = [num_nodes, 16] 
    dec_dims = [2*enc_dims[-1]*win_size, 32, latent_embedding_dim] 
    #enc_dims = [num_nodes, 16] # Layer configuration of encoder
    #dec_dims = [2*enc_dims[-1]*win_size, 32, num_nodes] # Layer configuration of decoder
    alpha = args.alpha
    beta = args.beta
    filter = args.filter

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)

    # ====================
    dropout_rate = args.dropout_rate # Dropout rate
    epsilon = 10 ** (-args.epsilon) # Threshold of zero-refining
    batch_size = args.batch_size # Batch size
    num_epochs = args.num_epochs # Number of training epochs
    num_val_snaps = args.num_val_snaps # Number of validation snapshots
    num_test_snaps = args.num_test_snaps # Number of test snapshots
    num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots
    lr_val = args.lr
    weight_decay_val = args.weight_decay

    print(f"data_name: {data_name}, max_thres: {max_thres}, win_size: {win_size}, "
      f"enc_dims: {enc_dims}, dec_dims: {dec_dims}, alpha: {alpha}, beta: {beta}, "
      f"dropout_rate: {dropout_rate}, epsilon: {epsilon}, batch_size: {batch_size}, "
      f"num_epochs: {num_epochs}, num_val_snaps: {num_val_snaps}, num_test_snaps: {num_test_snaps}, "
      f"num_train_snaps: {num_train_snaps}, lr_val: {lr_val}, weight_decay_val: {weight_decay_val}")

    print()
    # ====================
    # Define the model
    model = DDNE(enc_dims, dec_dims, dropout_rate,num_U=137, num_V=1218).to(device)
    num_U=137
    num_V=1218
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
        RMSE_list = []
        MAE_list = []
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
                neigh_tnr = neigh_tnr = torch.zeros((num_U, num_V)).to(device)
                for t in range(tau-win_size, tau):
                    edges = edge_seq[t]
                    adj = get_adj_wei_bipartite(edges, num_U, num_V, 137 , max_thres)
                    adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
                    adj_tnr = torch.FloatTensor(adj_norm).to(device)
                    adj_list.append(adj_tnr)
                    neigh_tnr += adj_tnr
                # ==========
                edges = edge_seq[tau]
                if filter:
                    edges = [e for e in edges if e[2] >= 0.1]
                gnd = get_adj_wei_bipartite(edges, num_U, num_V, 137 , max_thres) # Training ground-truth
                gnd_norm = gnd/max_thres  # Normalize the edge weights (in ground-truth) to [0, 1]
                gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
                # ==========
                adj_est, (emb_U, emb_V) = model(adj_list)
                combined_dyn_emb = torch.cat((emb_U, emb_V), dim=0) # Concatenate along the node dimension
                loss_ = get_DDNE_bipartite_loss(adj_est, gnd_tnr, neigh_tnr, combined_dyn_emb, alpha, beta, weight_class_1=5.0)

                #loss_ = get_DDNE_bipartite_loss(adj_est, gnd_tnr, neigh_tnr, emb_U, emb_V, alpha, beta)
                batch_loss = batch_loss + loss_
            # ==========
            # ===========================
            adj_est = adj_est.cpu().data.numpy() if torch.cuda.is_available() else adj_est.data.numpy()
            adj_est *= max_thres  # Rescale edge weights to the original value range

            # Calculate and store metrics
            RMSE = get_RMSE_(adj_est, gnd)
            MAE = get_MAE_(adj_est, gnd)            
            RMSE_list.append(RMSE)
            MAE_list.append(MAE)


            
            # Update model parameter according to batch loss
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total_loss = total_loss + batch_loss
            
        print('Epoch %d Total Loss %f' % (epoch, total_loss))
        RMSE_mean = np.mean(RMSE_list)
        RMSE_std = np.std(RMSE_list, ddof=1)
        MAE_mean = np.mean(MAE_list)
        MAE_std = np.std(MAE_list, ddof=1)

        print('Train Epoch %d RMSE %f %f MAE %f %f' % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std))


        # ====================
        # Validate the model
        model.eval()
        RMSE_list = []
        MAE_list = []

        precision_list = []
        recall_list = []

        for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
            # ====================
            adj_list = [] # List of historical adjacency matrices
            for t in range(tau-win_size, tau):
                # ==========
                edges = edge_seq[t]
                adj = get_adj_wei_bipartite(edges, num_U, num_V, 137 , max_thres)
                adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
                adj_tnr = torch.FloatTensor(adj_norm).to(device)
                adj_list.append(adj_tnr)
            # ====================
            # Get the prediction result
            adj_est, _ = model(adj_list)
            if torch.cuda.is_available():
                adj_est = adj_est.cpu().data.numpy()
            else:
                adj_est = adj_est.data.numpy()
            adj_est *= max_thres # Rescale edge weights to the original value range
            # ==========
            # Get ground-truth
            edges = edge_seq[tau]
            gnd = get_adj_wei_bipartite(edges, num_U, num_V, 137 , max_thres)
            # ====================
            # Evaluate the quality of current prediction operation
            RMSE = get_RMSE_(adj_est, gnd)
            MAE = get_MAE_(adj_est, gnd)
            RMSE_list.append(RMSE)
            MAE_list.append(MAE)
                # ==========
            # Clasificación binaria para métricas en clase 1
            threshold = 1.0
            pred_binary = (adj_est >= threshold).astype(int)
            gnd_binary = (gnd >= threshold).astype(int)

            # Flatten bipartite region (U x V)
            pred_flat = pred_binary[0:num_U, 137:].flatten()
            gnd_flat = gnd_binary[0:num_U, 137:].flatten()

            # Evitar advertencias si no hay positivos
            if np.sum(gnd_flat) > 0:
                precision = precision_score(gnd_flat, pred_flat, pos_label=1, zero_division=0)
                recall = recall_score(gnd_flat, pred_flat, pos_label=1, zero_division=0)
                precision_list.append(precision)
                recall_list.append(recall)

        # ====================
        RMSE_mean = np.mean(RMSE_list)
        RMSE_std = np.std(RMSE_list, ddof=1)
        MAE_mean = np.mean(MAE_list)
        MAE_std = np.std(MAE_list, ddof=1)
        precision_mean = np.mean(precision_list)
        recall_mean = np.mean(recall_list)



        print('Val Epoch %d RMSE %f %f MAE %f %f' % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std))
        print(f"Precision clase 1: {precision_mean:.4f}")
        print(f"Recall clase 1: {recall_mean:.4f}")

       
    # ====================
    predictions = []
    snapshot_indices = []
    classification_reports = []

    # Iterative Prediction over Test Years
    print("------- Iterative Prediction Test -------")
    start_test = num_snaps - num_test_snaps
    # Initialize current_window with real data
    current_window = []
    for t in range(start_test - win_size, start_test):
        edges = edge_seq[t]
        adj = get_adj_wei_bipartite(edges, num_U, num_V, 137 , max_thres)
        adj_norm = adj / max_thres
        current_window.append(torch.FloatTensor(adj_norm).to(device))


    # Iterate on test snapshots
    for tau in range(start_test, num_snaps):
        model.eval()
        with torch.no_grad():
            adj_est, _ = model(current_window)
        adj_est = (adj_est.cpu().data.numpy() if torch.cuda.is_available() 
                   else adj_est.data.numpy())
        adj_est *= max_thres
        predictions.append(adj_est)
        
        # Calculate metrics comparing them with ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_wei_bipartite(edges, num_U, num_V, 137 , max_thres)

        ### 
        #This part filters unwanted connections; we have a bipartite graph but DDNE takes it as a squared matrix, which makes a lot of noise in the results

        max_country = 136 #Max country index is 136
        min_product = 137  # Minimum product index is 137
        max_product = 1354 #max product index is 1354 
        total_nodes = max_product + 1  #so we have 1355 total nodes

        true_labels = (gnd >= 1).astype(int).flatten()
        pred_labels = (adj_est >= 1).astype(int).flatten()

        ###

        RMSE = get_RMSE_(adj_est, gnd)
        MAE = get_MAE_(adj_est, gnd)

        print(f"Iterative Prediction Test on year {tau - start_test + 1}: RMSE {RMSE}, MAE {MAE}")
        print()
        #print(f"Iterative Prediction Test on year {tau - start_test + 1}: Filtered RMSE: {filtered_rmse} std: {rmse_std},  MAE {filtered_mae}, std: {mae_std}")

        # Classification stats

        # Classification per snapshot
        class_report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

        
        snapshot_index = tau - start_test + 1
        snapshot_indices.append(snapshot_index)
        classification_reports.append(class_report)

        #print(f"Snapshot {snapshot_index}: AUC={roc_auc:.3f}, AUC-PR={avg_prec:.3f}, Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
        print("Classification report:")
        print(classification_report(true_labels, pred_labels, zero_division=0))

        # Update window: we pop the oldest snapshot and them we append the latest prediction
        current_window.pop(0)
        current_window.append(torch.FloatTensor((adj_est / max_thres)).to(device))
    
    print()
    print('Total runtime was: %s seconds' % (time.time() - start_time))

if __name__ == "__main__":
    main()
