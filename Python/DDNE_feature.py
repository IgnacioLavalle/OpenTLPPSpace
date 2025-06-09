# Demonstration of DDNE
import time
import torch
import torch.optim as optim
from DDNE.modules import *
from DDNE.loss import *
from utils import *
import argparse
import json
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report,
    precision_recall_curve, average_precision_score
)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of DDNE")
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


    return parser.parse_args()

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
    enc_dims = [num_nodes, 16] # Layer configuration of encoder
    dec_dims = [2*enc_dims[-1]*win_size, 32, num_nodes] # Layer configuration of decoder
    alpha = args.alpha
    beta = args.beta

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
    model = DDNE(enc_dims, dec_dims, dropout_rate).to(device)
    # ==========
    # Define the optimizer
    opt = optim.Adam(model.parameters(), lr=lr_val, weight_decay=weight_decay_val)
    
    ###
    max_country = 136 #Max country index is 136
    min_product = 137  # Minimum product index is 137
    max_product = 1354 #max product index is 1354 
    total_nodes = max_product + 1  #so we have 1355 total nodes
 
    valid_mask = np.zeros((total_nodes, total_nodes), dtype=bool)

    valid_mask[0:137, 137:1355] = True
    node_labels = np.zeros(num_nodes, dtype=np.float32)
    node_labels[137:] = 1.0
    node_labels_tnr = torch.FloatTensor(node_labels).unsqueeze(1).to(device)

    ###


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
                neigh_tnr = torch.zeros((num_nodes, num_nodes)).to(device)
                for t in range(tau-win_size, tau):
                    edges = edge_seq[t]
                    adj = get_adj_wei(edges, num_nodes, max_thres)
                    adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
                    adj_tnr = torch.FloatTensor(adj_norm).to(device)
                    adj_list.append(adj_tnr)
                    neigh_tnr += adj_tnr
                # ==========
                edges = edge_seq[tau]
                gnd = get_adj_wei(edges, num_nodes, max_thres) # Training ground-truth
                gnd_norm = gnd/max_thres  # Normalize the edge weights (in ground-truth) to [0, 1]
                gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
                # ==========

                # ===================================
                # ===================================
                adj_est, dyn_emb = model(adj_list)
                dyn_emb = torch.cat([dyn_emb, node_labels_tnr], dim=1)
                loss_ = get_DDNE_loss(adj_est, gnd_tnr, neigh_tnr, dyn_emb, alpha, beta)
                batch_loss = batch_loss + loss_
            # ==========
            adj_est = adj_est.cpu().data.numpy() if torch.cuda.is_available() else adj_est.data.numpy()
            adj_est *= max_thres  # Rescale edge weights to the original value range

            # Calculate and store metrics
            RMSE = get_RMSE(adj_est, gnd, num_nodes)
            MAE = get_MAE(adj_est, gnd, num_nodes)            
            RMSE_list.append(RMSE)
            MAE_list.append(MAE)


            
            # Update model parameter according to batch loss
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            adj_est, _ = model(adj_list)
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
       
    # ====================
    # Iterative Prediction over Test Years
    print("------- Iterative Prediction Test -------")
    start_test = num_snaps - num_test_snaps
    # Initialize current_window with real data
    current_window = []
    for t in range(start_test - win_size, start_test):
        edges = edge_seq[t]
        adj = get_adj_wei(edges, num_nodes, max_thres)
        adj_norm = adj / max_thres
        current_window.append(torch.FloatTensor(adj_norm).to(device))
    
    predictions = []
    snapshot_indices = []
    fpr_list = []
    tpr_list = []
    roc_auc_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    classification_reports = []
    precision_curve_list = []
    recall_curve_list = []
    average_precision_list = []



    # Iterate on test snapshots
    for tau in range(start_test, num_snaps):
        model.eval()
        with torch.no_grad():
            adj_est, _ = model(current_window)
        adj_est = (adj_est.cpu().data.numpy() if torch.cuda.is_available() 
                   else adj_est.data.numpy())
        adj_est *= max_thres
        # Prediction refinement
        adj_est = (adj_est + adj_est.T) / 2
        np.fill_diagonal(adj_est, 0)
        for r in range(num_nodes):
            for c in range(num_nodes):
                if adj_est[r, c] <= epsilon:
                    adj_est[r, c] = 0
        predictions.append(adj_est)
        
        # Calculate metrics comparing them with ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)

        ### 
        #This part filters unwanted connections; we have a bipartite graph but DDNE takes it as a squared matrix, which makes a lot of noise in the results


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
        kl = get_EW_KL(adj_est, gnd, num_nodes)

        print(f"Iterative Prediction Test on year {tau - start_test + 1}: RMSE {RMSE}, MAE {MAE}, KL {kl}")
        print()
        print(f"Iterative Prediction Test on year {tau - start_test + 1}: Filtered RMSE: {filtered_rmse} std: {rmse_std},  MAE {filtered_mae}, std: {mae_std}")

        # Classification stats

        # Classification per snapshot
        fpr, tpr, _ = roc_curve(true_labels, pred_scores)
        roc_auc = auc(fpr, tpr)
        acc = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels, zero_division=0)
        rec = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        class_report = classification_report(true_labels, pred_labels, output_dict=True)

        # Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(true_labels, pred_scores)
        avg_prec = average_precision_score(true_labels, pred_scores)

        precision_curve_list.append(precision_vals.tolist())
        recall_curve_list.append(recall_vals.tolist())
        average_precision_list.append(avg_prec)


        
        snapshot_index = tau - start_test + 1
        snapshot_indices.append(snapshot_index)
        fpr_list.append(fpr.tolist())
        tpr_list.append(tpr.tolist())
        roc_auc_list.append(roc_auc)
        accuracy_list.append(acc)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        classification_reports.append(class_report)

        print(f"Snapshot {snapshot_index}: AUC={roc_auc:.3f}, AUC-PR={avg_prec:.3f}, Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
        print("Classification report:")
        print(classification_report(true_labels, pred_labels))

        # Update window: we pop the oldest snapshot and them we append the latest prediction
        current_window.pop(0)
        current_window.append(torch.FloatTensor((adj_est / max_thres)).to(device))
    
    if save_forecast:
        filename_npy = f'predictionsWith_{num_train_snaps}Train_{num_val_snaps}Val_{num_test_snaps}TestSnaps.npy'
        np.save(filename_npy, np.array(predictions, dtype=object))

    if save_metrics:
        # save metrics as json
        metrics_summary = {
            "snapshots": snapshot_indices,
            "fpr": fpr_list,
            "tpr": tpr_list,
            "roc_auc": roc_auc_list,
            "accuracy": accuracy_list,
            "precision": precision_list,
            "recall": recall_list,
            "f1": f1_list,
            "classification_reports": classification_reports,
            "precision_curve": precision_curve_list,
            "recall_curve": recall_curve_list,
            "average_precision": average_precision_list

        }

        filename = f"snapshot_metrics_train{num_train_snaps}_test{num_test_snaps}.json"
        with open(filename, "w") as f:
            json.dump(metrics_summary, f, indent=2)


    print()
    print('Total runtime was: %s seconds' % (time.time() - start_time))

if __name__ == "__main__":
    main()
