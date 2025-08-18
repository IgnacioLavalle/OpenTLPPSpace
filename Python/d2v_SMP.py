# Demonstration of dyngraph2vec

import argparse
import time
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.optim as optim
from dyngraph2vec.modules import *
from dyngraph2vec.loss import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of d2v")
    #adding arguments and their respective default value

    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs (default: 500)")
    parser.add_argument("--num_val_snaps", type=int, default=3, help="Number of validation snapshots (default: 3)")
    parser.add_argument("--num_test_snaps", type=int, default=3, help="Number of test snapshots (default: 3)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay (default: 5e-4)")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta value (default: 0.2)")
    parser.add_argument("--win_size", type=int, default=2, help="Window size of historical snapshots (default: 2)")
    parser.add_argument("--max_thres", type=float, default=2.0, help="Threshold for maximum edge weight (default: 2) (el maximo del grafo es 17500)")
    parser.add_argument("--data_name", type=str, default ='SMP22to95', help = "Dataset name")


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
    max_thres = args.max_thres # Threshold for maximum edge weight
    struc_dims = [num_nodes, 32] # Layer configuration of structural encoder (FC)
    temp_dims = [struc_dims[-1], 16, 16] # Layer configuration of temporal encoder (RNN)
    dec_dims = [temp_dims[-1], 32, num_nodes] # Layer configuration of decoder (FC)
    beta = args.beta # Hyper-parameter of loss

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)

    # ====================
    dropout_rate = args.dropout_rate # Dropout rate
    win_size = args.win_size # Window size of historical snapshots
    batch_size = args.batch_size # Batch size
    num_epochs = args.num_epochs # Number of training epochs
    num_val_snaps = args.num_val_snaps # Number of validation snapshots
    num_test_snaps = args.num_test_snaps # Number of test snapshots
    num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots
    lr_val = args.lr
    wdecay = args.weight_decay

    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True

    best_val_f1 = -1.0 # Or any metric you want to track for 'best' model
    best_epoch = -1

    print(f"data_name: {data_name}, max_thres: {max_thres}, win_size: {win_size}, "
      f"dropout_rate: {dropout_rate}, beta: {beta}, batch_size: {batch_size}, "
      f"num_epochs: {num_epochs}, num_val_snaps: {num_val_snaps}, num_test_snaps: {num_test_snaps}, "
      f"num_train_snaps: {num_train_snaps}, lr_val: {lr_val}, weight_decay_val: {wdecay}")

    print()


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
            pred_labels = (pred_vals >= 1).astype(int)

            append_classification_metrics_with(c0precision_list, c0recall_list, c0f1_list, c1precision_list, c1recall_list, c1f1_list, true_labels, pred_labels)

            #Errors
            abs_errors = np.abs(pred_vals - true_vals)
            sq_errors = (pred_vals - true_vals) ** 2

            #MAE and std
            MAE = np.mean(abs_errors)

            #RMSE and std
            RMSE = np.sqrt(np.mean(sq_errors))

            # ==========
            RMSE_list.append(RMSE)
            MAE_list.append(MAE)

        # ====================
        RMSE_mean = np.mean(RMSE_list)
        RMSE_std = np.std(RMSE_list, ddof=1)
        MAE_mean = np.mean(MAE_list)
        MAE_std = np.std(MAE_list, ddof=1)

        # Classification metrics per class: Precision, Recall, F1
        c0_prec_mean, c0_prec_std, c1_prec_mean, c1_prec_std = mean_and_std_from_classlists(c0precision_list, c1precision_list)

        c0_recall_mean, c0_recall_std, c1_recall_mean, c1_recall_std = mean_and_std_from_classlists(c0recall_list, c1recall_list)

        c0_f1_mean, c0_f1_std, c1_f1_mean, c1_f1_std = mean_and_std_from_classlists(c0f1_list, c1f1_list)

        print('Val Pre Epoch %d RMSE %f %f MAE %f %f'
            % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std))
        print('  C0 Prec: %f (+-%f) C0 Rec: %f (+-%f) C0 F1: %f (+-%f)' %
            (c0_prec_mean, c0_prec_std,
            c0_recall_mean, c0_recall_std,
            c0_f1_mean, c0_f1_std))
        print('  C1 Prec: %f (+-%f) C1 Rec: %f (+-%f) C1 F1: %f (+-%f)' %
            (c1_prec_mean, c1_prec_std,
            c1_recall_mean, c1_recall_std,
            c1_f1_mean, c1_f1_std))
        
        if c1_f1_mean > best_val_f1:
            best_val_f1 = c1_f1_mean
            best_epoch = epoch

    # ====================
    # Test the model
    model.eval()
    RMSE_list = []
    MAE_list = []

    c0precision_list = []
    c0recall_list = []
    c0f1_list = []
    c1precision_list = []
    c1recall_list = []
    c1f1_list = []

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
        # ====================
        # Get the ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ====================
        # Evaluate the quality of current prediction operation
        true_vals = gnd[valid_mask]
        pred_vals = adj_est[valid_mask]

        true_labels = (true_vals >= 1).astype(int)
        pred_labels = (pred_vals >= 1).astype(int)

        print('Test snapshot %d metrics per snapshot:'
            % (tau))
        print()

        append_classification_metrics_with(c0precision_list, c0recall_list, c0f1_list, c1precision_list, c1recall_list, c1f1_list, true_labels, pred_labels)

        print('  C0 Prec: %f  C0 Rec: %f  C0 F1: %f' %
            (c0precision_list[-1],
            c0recall_list[-1],
            c0f1_list[-1]))
        print('  C1 Prec: %f  C1 Rec: %f  C1 F1: %f' %
            (c1precision_list[-1],
            c1recall_list[-1],
            c1f1_list[-1]))
        print()

        #Errors
        abs_errors = np.abs(pred_vals - true_vals)
        sq_errors = (pred_vals - true_vals) ** 2

        #MAE and std
        MAE = np.mean(abs_errors)

        #RMSE and std
        RMSE = np.sqrt(np.mean(sq_errors))

        # ==========
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
    # ====================
    
    # Classification metrics per class: Precision, Recall, F1
    c0_prec_mean, c0_prec_std, c1_prec_mean, c1_prec_std = mean_and_std_from_classlists(c0precision_list, c1precision_list)

    c0_recall_mean, c0_recall_std, c1_recall_mean, c1_recall_std = mean_and_std_from_classlists(c0recall_list, c1recall_list)

    c0_f1_mean, c0_f1_std, c1_f1_mean, c1_f1_std = mean_and_std_from_classlists(c0f1_list, c1f1_list)

    # Regression metrics: Rmse and Mae
    RMSE_mean = np.mean(RMSE_list)
    RMSE_std = np.std(RMSE_list, ddof=1)
    MAE_mean = np.mean(MAE_list)
    MAE_std = np.std(MAE_list, ddof=1)


    print('Test Epoch %d RMSE %f %f MAE %f %f'
        % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std))
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
    print('Best F1 during validation was: %f during epoch: %d' % (best_val_f1, best_epoch))

    print()
    print('Total runtime was: %s seconds' % (time.time() - start_time))

if __name__ == "__main__":
    main()
