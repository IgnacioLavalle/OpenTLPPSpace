# Demonstration of TMF

import argparse
import time
from sklearn.metrics import precision_recall_fscore_support
from TMF.TMF import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of TMF")
    #adding arguments and their respective default value

    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs (default: 500)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for generator (default: 1e-2)")
    parser.add_argument("--win_size", type=int, default=2, help="Window size of historical snapshots (default: 2)")
    parser.add_argument("--max_thres", type=float, default=2.0, help="Threshold for maximum edge weight (default: 1) (el maximo del grafo es 17500)")
    parser.add_argument("--data_name", type=str, default ='SMP22to95unweighted', help = "Dataset name")
    parser.add_argument("--theta", type=float, default=0.1, help="theta value (default: 0.1)")
    parser.add_argument("--beta", type=float, default=0.01, help="beta value (default: 0.01)")
    parser.add_argument("--alpha", type=float, default=0.01, help="alpha value (default: 0.01)")
    parser.add_argument("--hid_dim", type = int, default=16, help="Dimensionality of latent space")

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
    data_name = args.data_name
    num_nodes = 1355 # Number of nodes (Level-1 w/ fixed node set)
    num_snaps = 28 # Number of snapshots
    hid_dim = args.hid_dim # Dimensionality of latent space
    theta = args.theta
    alpha = args.alpha
    beta = args.beta

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True
    # ====================
    learn_rate = args.lr
    win_size = args.win_size # Window size of historical snapshots
    num_epochs = args.num_epochs # Number of training epochs

    print(f"data_name: {data_name}, win_size: {win_size}, "
      f"hid_dim: {hid_dim}, alpha: {alpha}, theta: {theta}, beta: {beta}, "
      f"num_epochs: {num_epochs}, lr: {learn_rate}")
    print()

    # ====================
    c0precision_list = []
    c0recall_list = []
    c0f1_list = []
    c1precision_list = []
    c1recall_list = []
    c1f1_list = []

    for tau in range(win_size, num_snaps):
        edges = edge_seq[tau]
        gnd = get_adj_unweighted(edges, num_nodes)
        # ==========
        adj_list = [] # List of historical adjacency matrices
        for t in range(tau-win_size, tau):
            edges = edge_seq[t]
            adj = get_adj_unweighted(edges, num_nodes)
            adj = adj
            adj_tnr = torch.FloatTensor(adj).to(device)
            adj_list.append(adj_tnr)
        TMF_model = TMF(num_nodes, hid_dim, win_size, num_epochs, alpha, beta, theta, learn_rate, device)
        adj_est = TMF_model.TMF_fun(adj_list)
        adj_est = adj_est
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()

        # ==========
        # Evaluate the quality of current prediction operation

        true_vals = gnd[valid_mask]
        pred_vals = adj_est[valid_mask]

        true_labels = (true_vals >= 1).astype(int)
        pred_labels = (pred_vals >= 1).astype(int)

        append_classification_metrics_with(c0precision_list, c0recall_list, c0f1_list, c1precision_list, c1recall_list, c1f1_list, true_labels, pred_labels)
        print('snapshot %d metrics' %(tau))
        print()
        print('  C0 Prec: %f  C0 Rec: %f  C0 F1: %f' %
            (c0precision_list[-1],
            c0recall_list[-1],
            c0f1_list[-1]))
        print('  C1 Prec: %f  C1 Rec: %f  C1 F1: %f' %
            (c1precision_list[-1],
            c1recall_list[-1],
            c1f1_list[-1]))
        print()


    # ====================

    # Classification metrics per class: Precision, Recall, F1
    c0_prec_mean, c0_prec_std, c1_prec_mean, c1_prec_std = mean_and_std_from_classlists(c0precision_list, c1precision_list)

    c0_recall_mean, c0_recall_std, c1_recall_mean, c1_recall_std = mean_and_std_from_classlists(c0recall_list, c1recall_list)

    c0_f1_mean, c0_f1_std, c1_f1_mean, c1_f1_std = mean_and_std_from_classlists(c0f1_list, c1f1_list)
    
    print('Final metrics mean and std')
    print()
    print('  C0 Prec: %f (+-%f) C0 Rec: %f (+-%f) C0 F1: %f (+-%f)' %
        (c0_prec_mean, c0_prec_std,
        c0_recall_mean, c0_recall_std,
        c0_f1_mean, c0_f1_std))
    print('  C1 Prec: %f (+-%f) C1 Rec: %f (+-%f) C1 F1: %f (+-%f)' %
        (c1_prec_mean, c1_prec_std,
        c1_recall_mean, c1_recall_std,
        c1_f1_mean, c1_f1_std))
  
    print()
    print('Total runtime was: %s seconds' % (time.time() - start_time))
    
if __name__ == '__main__':
    main()