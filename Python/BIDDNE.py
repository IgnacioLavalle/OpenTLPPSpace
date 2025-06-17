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

    # Parámetros generales
    data_name = 'SMP22to95'
    num_nodes = 1355
    num_snaps = 28
    max_thres = args.max_thres
    win_size = args.win_size
    latent_embedding_dim = 128
    enc_dims = enc_dims = [1218, 16]
    dec_dims = [2 * enc_dims[-1] * win_size, 32, latent_embedding_dim]
    num_U =137
    num_V = 1218

    alpha, beta = args.alpha, args.beta
    dropout_rate = args.dropout_rate
    epsilon = 10 ** (-args.epsilon)
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_val_snaps = args.num_val_snaps
    num_test_snaps = args.num_test_snaps
    num_train_snaps = num_snaps - num_val_snaps - num_test_snaps
    lr_val = args.lr
    weight_decay_val = args.weight_decay
    filter_edges = args.filter
    weight_class_1 = args.wc

    print(f"data_name: {data_name}, max_thres: {max_thres}, win_size: {win_size}, "
          f"enc_dims: {enc_dims}, dec_dims: {dec_dims}, alpha: {alpha}, beta: {beta}, "
          f"dropout_rate: {dropout_rate}, epsilon: {epsilon}, batch_size: {batch_size}, "
          f"num_epochs: {num_epochs}, num_val_snaps: {num_val_snaps}, num_test_snaps: {num_test_snaps}, "
          f"num_train_snaps: {num_train_snaps}, lr_val: {lr_val}, weight_decay_val: {weight_decay_val}")
    print()

    # Cargar datos
    edge_seq = np.load(f'data/{data_name}_edge_seq.npy', allow_pickle=True)

    # Definir modelo y optimizador
    model = DDNE(enc_dims, dec_dims, dropout_rate, num_u=137, num_v=1218).to(device)
    opt = optim.Adam(model.parameters(), lr=lr_val, weight_decay=weight_decay_val)

    # Entrenamiento
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        RMSE_list, MAE_list = [], []

        for b in range(int(np.ceil(num_train_snaps / batch_size))):
            start_idx, end_idx = b * batch_size, min((b + 1) * batch_size, num_train_snaps)
            batch_loss = 0.0

            for tau in range(start_idx, end_idx):
                adj_list = []
                neigh_tnr = torch.zeros((num_U, num_V)).to(device)

                for t in range(tau - win_size, tau):
                    adj = get_adj_wei_bipartite(edge_seq[t], num_U, num_V, num_U, max_thres)
                    adj_tnr = torch.FloatTensor(adj / max_thres).to(device)
                    adj_list.append(adj_tnr)
                    neigh_tnr += adj_tnr

                edges = edge_seq[tau]
                if filter_edges:
                    edges = [e for e in edges if e[2] >= 0.1]
                gnd = get_adj_wei_bipartite(edges, num_U, num_V, num_U, max_thres)
                gnd_tnr = torch.FloatTensor(gnd / max_thres).to(device)

                adj_est, (emb_U, emb_V) = model(adj_list)
                loss_ = get_DDNE_bipartite_loss(adj_est, gnd_tnr, emb_U, emb_V, beta=alpha, weight_class_1=weight_class_1)
                batch_loss += loss_

                # Métricas para esta iteración
                adj_est_np = adj_est.detach().cpu().numpy() * max_thres
                RMSE_list.append(get_RMSE_(adj_est_np, gnd))
                MAE_list.append(get_MAE_(adj_est_np, gnd))

            # Optimización
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += batch_loss

        print(f"Epoch {epoch} Total Loss: {total_loss:.4f}")
        print(f"Train Epoch {epoch} RMSE: {np.mean(RMSE_list):.4f} ± {np.std(RMSE_list, ddof=1):.4f} "
              f"MAE: {np.mean(MAE_list):.4f} ± {np.std(MAE_list, ddof=1):.4f}")

        # Validación
        model.eval()
        RMSE_list, MAE_list = [], []
        precision_list, recall_list = [], []

        for tau in range(num_snaps - num_test_snaps - num_val_snaps, num_snaps - num_test_snaps):
            adj_list = []
            for t in range(tau - win_size, tau):
                adj = get_adj_wei_bipartite(edge_seq[t], num_U, num_V, num_U, max_thres)
                adj_tnr = torch.FloatTensor(adj / max_thres).to(device)
                adj_list.append(adj_tnr)

            with torch.no_grad():
                adj_est, _ = model(adj_list)
            adj_est_np = adj_est.cpu().numpy() * max_thres
            gnd = get_adj_wei_bipartite(edge_seq[tau], num_U, num_V, num_U, max_thres)

            RMSE_list.append(get_RMSE_(adj_est_np, gnd))
            MAE_list.append(get_MAE_(adj_est_np, gnd))

            # Clasificación binaria para métricas
            pred_bin = (adj_est_np >= 1.0).astype(int)
            gnd_bin = (gnd >= 1.0).astype(int)

            pred_flat = pred_bin[0:num_U, num_U:].flatten()
            gnd_flat = gnd_bin[0:num_U, num_U:].flatten()

            if np.sum(gnd_flat) > 0:
                precision_list.append(precision_score(gnd_flat, pred_flat, zero_division=0))
                recall_list.append(recall_score(gnd_flat, pred_flat, zero_division=0))

        print(f"Val Epoch {epoch} RMSE: {np.mean(RMSE_list):.4f} ± {np.std(RMSE_list, ddof=1):.4f} "
              f"MAE: {np.mean(MAE_list):.4f} ± {np.std(MAE_list, ddof=1):.4f}")
        print(f"Precision clase 1: {np.mean(precision_list):.4f}")
        print(f"Recall clase 1: {np.mean(recall_list):.4f}")

    # ... continuación con predicciones iterativas ...


       
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
