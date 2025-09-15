# Demonstration of STGSN

import argparse
import json
import time
import torch
import torch.optim as optim
from STGSN.modules import *
from STGSN.loss import *
from utils import *
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, precision_recall_fscore_support, roc_curve, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of STSGN")
    #adding arguments and their respective default value

    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--epsilon", type=int, default=2, help="Threshold of zero-refining (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs (default: 100)")
    parser.add_argument("--num_test_snaps", type=int, default=6, help="Number of test snapshots (default: 3)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (default: 1e-4)")
    parser.add_argument("--theta", type=float, default=0.1, help="theta value (default: 0.1)")
    parser.add_argument("--win_size", type=int, default=2, help="Window size of historical snapshots (default: 2)")
    parser.add_argument("--max_thres", type=float, default=2.0, help="Threshold for maximum edge weight (default: 2) (el maximo del grafo es 17500)")
    parser.add_argument("--data_name", type=str, default ='SMP22to95', help = "Dataset name")
    parser.add_argument("--feat_name", type=str, default ='SMP22to95_oh', help = "Dataset features name")
    parser.add_argument("--feat_dim", type=int, default=32, help="dimensionality of feautre input (default: 32)")
    parser.add_argument("--enc_dim", type=int, default=32, help="dimensionality of enconder (default: 32)")
    parser.add_argument("--save_metrics", type=bool, default=False)



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
    # ====================
    f_layer = s_layer = t_layer = args.enc_dim
    #If you choose enconder dimension to be the same you have [feat_dim, f_layer,f_layer,f_layer]
    #So if instead you choose encoder dimensions to be different, you have [feat_dim, f_layer,f_layer * 2,f_layer * 4]
    s_layer *= 2
    t_layer *= 4

    data_name = args.data_name
    feat_name = args.feat_name
    num_nodes = 1355 # Number of nodes (Level-1 w/ fixed node set)
    num_snaps = 28 # Number of snapshots
    max_thres = args.max_thres # Threshold for maximum edge weight
    feat_dim = args.feat_dim # Dimensionality of feature input
    enc_dims = [feat_dim, f_layer, s_layer, t_layer] # Layer configuration of encoder
    emb_dim = enc_dims[-1] # Dimensionality of dynamic embedding
    win_size = args.win_size # Window size of historical snapshots
    theta = args.theta # Hyper-parameter for collapsed graph
    save_metrics = args.save_metrics

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
    feat = np.load('data/%s_feat.npy' % (feat_name), allow_pickle=True)
    
    feat_tnr = torch.FloatTensor(feat).to(device)

    feat_list = []
    for i in range(win_size):
        feat_list.append(feat_tnr)

    # ====================
    dropout_rate = args.dropout_rate # Dropout rate
    batch_size = args.batch_size # Batch size
    num_epochs = args.num_epochs # Number of training epochs
    num_val_snaps = args.num_val_snaps # Number of validation snapshots
    num_test_snaps = args.num_test_snaps # Number of test snapshots
    num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots
    lr = args.lr
    weight_decay = args.weight_decay

    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True

    best_val_f1 = -1.0 # Or any metric you want to track for 'best' model
    best_epoch = -1


    print(f"data_name: {data_name}, max_thres: {max_thres}, win_size: {win_size}, "
      f"enc_dims: {enc_dims}, theta: {theta}, "
      f"dropout_rate: {dropout_rate},  batch_size: {batch_size}, "
      f"num_epochs: {num_epochs}, num_val_snaps: {num_val_snaps}, num_test_snaps: {num_test_snaps}, "
      f"num_train_snaps: {num_train_snaps}, lr_val: {lr}, weight_decay_val: {weight_decay}")

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
    # Iterative Forecast Test
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


    print("\n------- Iterative Forecast Test -------")

    start_test = num_snaps - num_test_snaps

    # Initialize windows with real snapshots
    current_edges = []
    for t in range(start_test - win_size, start_test):
        edges = edge_seq[t]
        adj = get_adj_wei(edges, num_nodes, max_thres)
        adj_norm = adj / max_thres
        current_edges.append(adj_norm)

    for tau in range(start_test, num_snaps):
        model.eval()
        with torch.no_grad():

            sup_list = []
            col_net = np.zeros((num_nodes, num_nodes))
            coef_sum = 0.0
            for i, adj_norm in enumerate(current_edges):
                sup = get_gnn_sup_d(adj_norm)
                sup_sp = sp.sparse.coo_matrix(sup)
                sup_sp = sparse_to_tuple(sup_sp)
                idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
                vals = torch.FloatTensor(sup_sp[1]).to(device)
                sup_tnr = torch.sparse_coo_tensor(idxs.t(), vals, torch.Size(sup_sp[2]),
                                                dtype=torch.float32, device=device)
                sup_list.append(sup_tnr)

                coef = (1-theta)**(win_size - i)
                col_net += coef * adj_norm
                coef_sum += coef
            col_net /= coef_sum

            col_sup = get_gnn_sup_d(col_net)
            col_sup_sp = sp.sparse.coo_matrix(col_sup)
            col_sup_sp = sparse_to_tuple(col_sup_sp)
            idxs = torch.LongTensor(col_sup_sp[0].astype(float)).to(device)
            vals = torch.FloatTensor(col_sup_sp[1]).to(device)
            col_sup_tnr = torch.sparse_coo_tensor(idxs.t(), vals, torch.Size(col_sup_sp[2]),
                                                dtype=torch.float32, device=device)

          
            adj_est = model(sup_list, feat_list, col_sup_tnr, feat_tnr, num_nodes)

        adj_est = adj_est.cpu().data.numpy() if torch.cuda.is_available() else adj_est.data.numpy()
        adj_est *= max_thres
        adj_est = (adj_est + adj_est.T) / 2  

        # Update window with current prediction
        adj_norm_pred = adj_est / max_thres
        current_edges.pop(0)
        current_edges.append(adj_norm_pred)

        # ====================
        # Evaluación
        gnd_edges = edge_seq[tau]
        gnd = get_adj_wei(gnd_edges, num_nodes, max_thres)

        true_vals = gnd[valid_mask]
        pred_vals = adj_est[valid_mask]

        true_labels = (true_vals >= 1).astype(int)
        pred_scores = pred_vals
        pred_labels = (pred_vals >= 1).astype(int)

        mask_class_1 = (true_labels == 1)
        mask_class_0 = (true_labels == 0)

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

        print(f"Iterative STGSN Prediction on snapshot {tau}:")
        print(f"  C0 Prec: {c0precision_list[-1]:.4f}  C0 Rec: {c0recall_list[-1]:.4f}  C0 F1: {c0f1_list[-1]:.4f}")
        print(f"  C1 Prec: {c1precision_list[-1]:.4f}  C1 Rec: {c1recall_list[-1]:.4f}  C1 F1: {c1f1_list[-1]:.4f}")
        print(f"  RMSE: {RMSE:.4f}  MAE: {MAE:.4f}")
        print(f"  Class 0 -> MAE: {mae_c0:.4f}  RMSE: {rmse_c0:.4f}")
        print(f"  Class 1 -> MAE: {mae_c1:.4f}  RMSE: {rmse_c1:.4f}\n")

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

        filename = f"stgsn_fc_curve_metrics_from_year{start_test}.json"
        with open(filename, "w") as f:
            json.dump(metrics_summary, f, indent=2)


if __name__ == "__main__":
    main()
