# Demonstration of LIST

import argparse
import json
import time

from sklearn.metrics import auc, average_precision_score, precision_recall_curve, precision_recall_fscore_support, roc_curve, accuracy_score
from LIST.LIST import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of GCNGAN")
    #adding arguments and their respective default value

    parser.add_argument("--num_epochs", type=int, default=750, help="Number of training epochs (default: 500)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for generator (default: 1e-4)")
    parser.add_argument("--win_size", type=int, default=3, help="Window size of historical snapshots (default: 2)")
    parser.add_argument("--data_name", type=str, default ='SMP22to95', help = "Dataset name")
    parser.add_argument("--theta", type=float, default=5.0, help="theta value (default: 5)")
    parser.add_argument("--beta", type=float, default=0.01, help="beta value (default: 0.01)")
    parser.add_argument("--lambd", type=float, default=0.1, help="theta value (default: 0.1)")
    parser.add_argument("--b", type= int, default=100, help="Number of iterations in order to get regularization matrix")
    parser.add_argument("--hid_dim", type = int, default=256, help="Dimensionality of latent space")
    parser.add_argument("--start", type = int, default=22, help="Forecast first year")
    parser.add_argument("--save_metrics", type=bool, default=False, help="Indicates whether you want or not to save the roc auc and pr auc metrics as json")


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
    save_metrics = args.save_metrics

    #Parameters
    data_name = args.data_name
    num_nodes = 1355
    num_snaps = 28
    hid_dim = args.hid_dim
    theta = args.theta
    beta = args.beta
    lambd = args.lambd
    b_iterations = args.b
    learn_rate = args.lr
    win_size = args.win_size
    num_epochs = args.num_epochs
    dec_list = get_dec_list(win_size, theta)

    edge_seq = np.load(f"data/{data_name}_edge_seq.npy", allow_pickle=True)
    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True

    print(f"data_name: {data_name}, win_size: {win_size}, "
          f"hid_dim: {hid_dim}, lambd: {lambd}, theta: {theta}, beta: {beta}, "
          f"b_iterations: {b_iterations}, num_epochs: {num_epochs}, lr: {learn_rate}\n")

    # === iterative config ===
    start_test = args.start  #Initial forecast snapshot 
    current_window = []

    #Initialize windows with ground truth before start_test
    for t in range(start_test - win_size, start_test):
        edges = edge_seq[t]
        adj = get_adj_unweighted(edges, num_nodes)
        adj_tnr = torch.FloatTensor(adj).to(device)
        current_window.append(adj_tnr)

    # Metric lists
    c0precision_list, c0recall_list, c0f1_list = [], [], []
    c1precision_list, c1recall_list, c1f1_list = [], [], []

    fpr_list = []
    tpr_list = []
    roc_auc_list = []
    accuracy_list = []


    precision_curve_list = []
    recall_curve_list = []
    average_precision_list = []

    # === Forecast ===
    for tau in range(start_test, num_snaps):
        # Generarate P_list from current window
        P_list = []
        for adj_tnr in current_window:
            adj_np = adj_tnr.cpu().numpy()
            P = get_P(adj_np, num_nodes, lambd, b_iterations, device=device)
            P_list.append(P)

        #LIST Model
        LIST_model = LIST(num_nodes, hid_dim, win_size, dec_list, P_list,
                          num_epochs, beta, learn_rate, device)
        adj_est = LIST_model.LIST_fun(current_window)
        adj_est = torch.nan_to_num(adj_est, nan=0.0, posinf=2.0, neginf=0.0)
        adj_est = adj_est.detach().cpu().numpy()


        # === Evaluate ===
        gnd_edges = edge_seq[tau]
        gnd = get_adj_unweighted(gnd_edges, num_nodes)

        true_vals = gnd[valid_mask]
        pred_vals = adj_est[valid_mask]

        true_labels = (true_vals >= 1).astype(int)
        pred_scores = pred_vals
        pred_labels = (pred_vals >= 1).astype(int)

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

        print(f"Iterative LIST Prediction on snapshot {tau}:")
        print(f"  C0 Prec: {c0precision_list[-1]:.4f}  C0 Rec: {c0recall_list[-1]:.4f}  C0 F1: {c0f1_list[-1]:.4f}")
        print(f"  C1 Prec: {c1precision_list[-1]:.4f}  C1 Rec: {c1recall_list[-1]:.4f}  C1 F1: {c1f1_list[-1]:.4f}")

        # Update window for next iteration
        adj_tnr_new = torch.FloatTensor(adj_est).to(device)
        current_window.pop(0)
        current_window.append(adj_tnr_new)

    # === Final metrics ===

    c0_prec_mean, c0_prec_std, c1_prec_mean, c1_prec_std = mean_and_std_from_classlists(c0precision_list, c1precision_list)
    c0_recall_mean, c0_recall_std, c1_recall_mean, c1_recall_std = mean_and_std_from_classlists(c0recall_list, c1recall_list)
    c0_f1_mean, c0_f1_std, c1_f1_mean, c1_f1_std = mean_and_std_from_classlists(c0f1_list, c1f1_list)

    print("Final metrics mean and std\n")
    print(f"  C0 Prec: {c0_prec_mean:.4f} (+-{c0_prec_std:.4f})  C0 Rec: {c0_recall_mean:.4f} (+-{c0_recall_std:.4f})  C0 F1: {c0_f1_mean:.4f} (+-{c0_f1_std:.4f})")
    print(f"  C1 Prec: {c1_prec_mean:.4f} (+-{c1_prec_std:.4f})  C1 Rec: {c1_recall_mean:.4f} (+-{c1_recall_std:.4f})  C1 F1: {c1_f1_mean:.4f} (+-{c1_f1_std:.4f})\n")
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

        filename = f"listfc_unweighted_curve_metrics_from_year{start_test}.json"
        with open(filename, "w") as f:
            json.dump(metrics_summary, f, indent=2)



if __name__ == '__main__':
    main()
