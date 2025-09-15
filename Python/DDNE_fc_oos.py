# Demonstration of DDNE
import time
import torch
import torch.optim as optim
from DDNE.modules import *
from DDNE.loss import *
from utils import *
import argparse
import json
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, precision_recall_fscore_support, roc_curve, accuracy_score




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of DDNE")
    #adding arguments and their respective default value

    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--epsilon", type=int, default=2, help="Threshold of zero-refining (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs (default: 100)")
    parser.add_argument("--num_test_snaps", type=int, default=6, help="Number of test snapshots (default: 3)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (default: 1e-4)")
    parser.add_argument("--alpha", type=float, default=3.0, help="Alpha value (default: 2.0)")
    parser.add_argument("--beta", type=float, default=0.0, help="Alpha value (default: 0.2)")
    parser.add_argument("--win_size", type=int, default=2, help="Window size of historical snapshots (default: 2)")
    parser.add_argument("--max_thres", type=float, default=2.0, help="Threshold for maximum edge weight (default: 1) (el maximo del grafo es 17500)")
    parser.add_argument("--save_forecast", type=bool, default=False, help="Indicates whether you want or not to save the forecast result")
    parser.add_argument("--save_metrics", type=bool, default=False, help="Indicates whether you want or not to save the classification metrics json")
    parser.add_argument("--data_name", type=str, default ='SMP22to95', help = "Dataset name")
    parser.add_argument("--hid_dim", type=int, default=256)


    return parser.parse_args()

def append_classification_metrics_with(c0precision_list, c0recall_list, c0f1_list, c1precision_list, c1recall_list, c1f1_list, true_labels, pred_labels):
    precision_per_class, recall_per_class, f1_per_class, _ = \
            precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0, 1], zero_division=0)
    c0precision_list.append(precision_per_class[0])
    c1precision_list.append(precision_per_class[1])
    c0recall_list.append(recall_per_class[0])
    c1recall_list.append(recall_per_class[1])
    c0f1_list.append(f1_per_class[0])
    c1f1_list.append(f1_per_class[1])


def mean_and_std_from_classlists(c0_list, c1_list):
    c0_mean = np.mean(c0_list)
    c0_std = np.std(c0_list, ddof=1) if len(c0_list) > 1 else 0.0
    c1_mean = np.mean(c1_list)
    c1_std = np.std(c1_list, ddof=1) if len(c1_list) > 1 else 0.0
    return c0_mean,c0_std,c1_mean,c1_std


def main():
    start_time = time.time()
    args = parse_args()

    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True

    save_forecast = args.save_forecast
    save_metrics = args.save_metrics
    # ====================
    data_name = args.data_name
    num_nodes = 1355 # Number of nodes (Level-1 w/ fixed node set)
    num_snaps = 28 # Number of snapshots
    max_thres = args.max_thres # Threshold for maximum edge weight
    win_size = args.win_size # Window size of historical snapshots
    h_dim = args.hid_dim
    t_dim = h_dim*2
    enc_dims = [num_nodes, h_dim] # Layer configuration of encoder
    dec_dims = [2*enc_dims[-1]*win_size, t_dim, num_nodes] # Layer configuration of decoder
    alpha = args.alpha
    beta = args.beta

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)

    # ====================
    dropout_rate = args.dropout_rate # Dropout rate
    epsilon = 10 ** (-args.epsilon) # Threshold of zero-refining
    batch_size = args.batch_size # Batch size
    num_epochs = args.num_epochs # Number of training epochs
    num_test_snaps = args.num_test_snaps # Number of test snapshots
    num_train_snaps = num_snaps-num_test_snaps # Number of training snapshots
    lr_val = args.lr
    weight_decay_val = args.weight_decay

    print(f"data_name: {data_name}, max_thres: {max_thres}, win_size: {win_size}, "
      f"enc_dims: {enc_dims}, dec_dims: {dec_dims}, alpha: {alpha}, beta: {beta}, "
      f"dropout_rate: {dropout_rate}, epsilon: {epsilon}, batch_size: {batch_size}, "
      f"num_epochs: {num_epochs}, num_test_snaps: {num_test_snaps}, "
      f"num_train_snaps: {num_train_snaps}, lr_val: {lr_val}, weight_decay_val: {weight_decay_val}")

    print()
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
                adj_est, dyn_emb = model(adj_list)
                loss_ = get_DDNE_loss(adj_est, gnd_tnr, neigh_tnr, dyn_emb, alpha, beta)
                batch_loss = batch_loss + loss_
            # ==========
            # ===========================
            adj_est = adj_est.cpu().data.numpy() if torch.cuda.is_available() else adj_est.data.numpy()
            adj_est *= max_thres  # Rescale edge weights to the original value range
            
            # Update model parameter according to batch loss
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total_loss = total_loss + batch_loss
            
        print('Epoch %d Total Loss %f' % (epoch, total_loss.detach().item()))

        """
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
       """
    # ====================

    predictions = []
    
    RMSE_list, MAE_list = [], []
    c0precision_list, c0recall_list, c0f1_list = [], [], []
    c1precision_list, c1recall_list, c1f1_list = [], [], []

    mae_c0_list, mae_c1_list = [], []
    rmse_c0_list, rmse_c1_list = [], []

    fpr_list = []
    tpr_list = []
    roc_auc_list = []
    accuracy_list = []


    precision_curve_list = []
    recall_curve_list = []
    average_precision_list = []

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

        predictions.append(adj_est)
        
        # Calculate metrics comparing them with ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)

        ### 
        #This part filters unwanted connections; we have a bipartite graph but DDNE takes it as a squared matrix, which makes a lot of noise in the results

        """
        max_country = 136 #Max country index is 136
        min_product = 137  # Minimum product index is 137
        max_product = 1354 #max product index is 1354 
        total_nodes = max_product + 1  #so we have 1355 total nodes
        """

        true_vals = gnd[valid_mask]
        pred_vals = adj_est[valid_mask]

        true_labels = (true_vals >= 1).astype(int)
        pred_scores = pred_vals
        pred_labels = (pred_vals >= 1).astype(int)
        mask_class_1 = (true_labels == 1)
        mask_class_0 = (true_labels == 0)

        #Errors
        abs_errors = np.abs(pred_vals - true_vals)
        sq_errors = (pred_vals - true_vals) ** 2

        #global Mae & rmse
        MAE = np.mean(abs_errors)
        RMSE = np.sqrt(np.mean(sq_errors))
        MAE_list.append(MAE)
        RMSE_list.append(RMSE)

        # MAE & RMSE per class
        mae_c1 = np.mean(abs_errors[mask_class_1])
        rmse_c1 = np.sqrt(np.mean(sq_errors[mask_class_1]))
        mae_c0 = np.mean(abs_errors[mask_class_0])
        rmse_c0 = np.sqrt(np.mean(sq_errors[mask_class_0]))

        mae_c1_list.append(mae_c1)
        rmse_c1_list.append(rmse_c1)
        mae_c0_list.append(mae_c0)
        rmse_c0_list.append(rmse_c0)

        #classification metrics
        append_classification_metrics_with(
            c0precision_list, c0recall_list, c0f1_list,
            c1precision_list, c1recall_list, c1f1_list,
            true_labels, pred_labels
        )
        # Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(true_labels, pred_scores)
        avg_prec = average_precision_score(true_labels, pred_scores)

        precision_curve_list.append(precision_vals.tolist())
        recall_curve_list.append(recall_vals.tolist())
        average_precision_list.append(avg_prec)

        # Roc Auc and accuracy
        fpr, tpr, _ = roc_curve(true_labels, pred_scores)
        roc_auc = auc(fpr, tpr)
        acc = accuracy_score(true_labels, pred_labels)

        fpr_list.append(fpr.tolist())
        tpr_list.append(tpr.tolist())
        roc_auc_list.append(roc_auc)
        accuracy_list.append(acc)

        print(f"Iterative DDNE Prediction on snapshot {tau}:")
        print(f"  C0 Prec: {c0precision_list[-1]:.4f}  C0 Rec: {c0recall_list[-1]:.4f}  C0 F1: {c0f1_list[-1]:.4f}")
        print(f"  C1 Prec: {c1precision_list[-1]:.4f}  C1 Rec: {c1recall_list[-1]:.4f}  C1 F1: {c1f1_list[-1]:.4f}")
        print(f"  RMSE: {RMSE:.4f}  MAE: {MAE:.4f}")
        print(f"  Class 0 -> MAE: {mae_c0:.4f}  RMSE: {rmse_c0:.4f}")
        print(f"  Class 1 -> MAE: {mae_c1:.4f}  RMSE: {rmse_c1:.4f}\n")

        # Update window: we pop the oldest snapshot and them we append the latest prediction
        current_window.pop(0)
        current_window.append(torch.FloatTensor((adj_est / max_thres)).to(device))

        # ====================

    # ====================
    
    # === Final metrics ===
    RMSE_mean, RMSE_std = np.mean(RMSE_list), np.std(RMSE_list, ddof=1)
    MAE_mean, MAE_std = np.mean(MAE_list), np.std(MAE_list, ddof=1)
    mae_c0_mean, mae_c0_std = np.mean(mae_c0_list), np.std(mae_c0_list, ddof=1)
    mae_c1_mean, mae_c1_std = np.mean(mae_c1_list), np.std(mae_c1_list, ddof=1)
    rmse_c0_mean, rmse_c0_std = np.mean(rmse_c0_list), np.std(rmse_c0_list, ddof=1)
    rmse_c1_mean, rmse_c1_std = np.mean(rmse_c1_list), np.std(rmse_c1_list, ddof=1)

    c0_prec_mean, c0_prec_std, c1_prec_mean, c1_prec_std = mean_and_std_from_classlists(c0precision_list, c1precision_list)
    c0_recall_mean, c0_recall_std, c1_recall_mean, c1_recall_std = mean_and_std_from_classlists(c0recall_list, c1recall_list)
    c0_f1_mean, c0_f1_std, c1_f1_mean, c1_f1_std = mean_and_std_from_classlists(c0f1_list, c1f1_list)

    print("Final metrics mean and std\n")
    print(f"  C0 Prec: {c0_prec_mean:.4f} (+-{c0_prec_std:.4f})  C0 Rec: {c0_recall_mean:.4f} (+-{c0_recall_std:.4f})  C0 F1: {c0_f1_mean:.4f} (+-{c0_f1_std:.4f})")
    print(f"  C1 Prec: {c1_prec_mean:.4f} (+-{c1_prec_std:.4f})  C1 Rec: {c1_recall_mean:.4f} (+-{c1_recall_std:.4f})  C1 F1: {c1_f1_mean:.4f} (+-{c1_f1_std:.4f})\n")
    print(f"RMSE {RMSE_mean:.4f} (+-{RMSE_std:.4f})  MAE {MAE_mean:.4f} (+-{MAE_std:.4f})")
    print(f"Class 0 -> MAE {mae_c0_mean:.4f} (+-{mae_c0_std:.4f}), RMSE {rmse_c0_mean:.4f} (+-{rmse_c0_std:.4f})")
    print(f"Class 1 -> MAE {mae_c1_mean:.4f} (+-{mae_c1_std:.4f}), RMSE {rmse_c1_mean:.4f} (+-{rmse_c1_std:.4f})\n")
    print(f"Total runtime was: {time.time() - start_time:.2f} seconds")

    
    """
    if save_forecast:
        filename_npy = f'predictionsWith_{num_train_snaps}Train_{num_val_snaps}Val_{num_test_snaps}TestSnaps.npy'
        np.save(filename_npy, np.array(predictions, dtype=object))
    """

    if save_metrics:
        # save metrics as json
        metrics_summary = {
            "fpr": fpr_list,
            "tpr": tpr_list,
            "roc_auc": roc_auc_list,
            "accuracy": accuracy_list,
            "precision_curve": precision_curve_list,
            "recall_curve": recall_curve_list,
            "average_precision": average_precision_list
        }

        filename = f"ddne_fc_curve_metrics_from_year{start_test}.json"
        with open(filename, "w") as f:
            json.dump(metrics_summary, f, indent=2)


    print()
    print('Total runtime was: %s seconds' % (time.time() - start_time))

if __name__ == "__main__":
    main()
