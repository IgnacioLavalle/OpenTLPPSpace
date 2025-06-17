# Demonstration of DDNE
import time
import torch
import torch.optim as optim
from DDNE.modules import *
from DDNE.loss import *
from utils import *
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import warnings



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
    parser.add_argument("--data_name", type=str, default ='SMP22to95', help = "Dataset name")


    return parser.parse_args()

def main():
    warnings.filterwarnings("ignore")

    start_time = time.time()
    args = parse_args()
    # ====================
    #data_name = 'SMP22to95'
    data_name = args.data_name
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
    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True
    node_labels = np.zeros((num_nodes, 2), dtype=np.float32)
    node_labels[:137, 1] = 1.0
    node_labels[137:, 0] = 1.0 
    node_labels_tnr = torch.FloatTensor(node_labels).to(device)

    best_val_f1 = -1.0 # Or any metric you want to track for 'best' model
    best_epoch = -1
    best_model_state = None # To store the state_dict of the best model




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
                adj_list = [] # List of historical adjacency matrices
                neigh_tnr = torch.zeros((num_nodes, num_nodes)).to(device)
                for t in range(tau-win_size, tau):
                    edges = edge_seq[t]
                    adj = get_adj_unweighted(edges, num_nodes)
                    adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
                    adj_tnr = torch.FloatTensor(adj_norm).to(device)
                    adj_list.append(adj_tnr)
                    neigh_tnr += adj_tnr
                # ==========
                edges = edge_seq[tau]
                gnd = get_adj_unweighted(edges, num_nodes) # Training ground-truth
                gnd_norm = gnd/max_thres  # Normalize the edge weights (in ground-truth) to [0, 1]
                gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
                # ==========
                adj_est, dyn_emb = model(adj_list)
                dyn_emb = torch.cat([dyn_emb, node_labels_tnr], dim=1)
                loss_ = get_DDNE_loss(adj_est, gnd_tnr, neigh_tnr, dyn_emb, alpha, beta)
                batch_loss = batch_loss + loss_
            # ==========
            # ===========================
            adj_est = adj_est.cpu().data.numpy() if torch.cuda.is_available() else adj_est.data.numpy()
            adj_est *= max_thres  # Rescale edge weights to the original value range
            
            # Update model parameter according to batch loss
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                adj = get_adj_unweighted(edges, num_nodes)
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
            # ====================
            # Get ground-truth
            edges = edge_seq[tau]
            gnd = get_adj_unweighted(edges, num_nodes)
            true_vals = gnd[valid_mask]
            pred_vals = adj_est[valid_mask]
            # ====================
            # Evaluate the quality of current prediction operation

            true_labels = (true_vals >= 1).astype(int)
            pred_labels = (pred_vals >= 1).astype(int)
            precision_per_class, recall_per_class, f1_per_class, _ = \
            precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0, 1], zero_division=0)
            c0precision_list.append(precision_per_class[0])
            c1precision_list.append(precision_per_class[1])
            c0recall_list.append(recall_per_class[0])
            c1recall_list.append(recall_per_class[1])
            c0f1_list.append(f1_per_class[0])
            c1f1_list.append(f1_per_class[1])

            # ====================
            # Evaluate the quality of current prediction operation
            c0_prec_mean = np.mean(c0precision_list)
            c0_prec_std = np.std(c0precision_list, ddof=1)
            c1_prec_mean = np.mean(c1precision_list)
            c1_prec_std = np.std(c1precision_list, ddof=1)

            c0_recall_mean = np.mean(c0recall_list)
            c0_recall_std = np.std(c0recall_list, ddof=1)
            c1_recall_mean = np.mean(c1recall_list)
            c1_recall_std = np.std(c1recall_list, ddof=1)

            c0_f1_mean = np.mean(c0f1_list)
            c0_f1_std = np.std(c0f1_list, ddof=1)
            c1_f1_mean = np.mean(c1f1_list)
            c1_f1_std = np.std(c1f1_list, ddof=1)


        # ====================

        print('Val Epoch %d' % epoch)
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
            best_model_state = model.state_dict() # Save model's parameters
            print(f"  --> New best validation C1 F1: {best_val_f1:.4f} at epoch {best_epoch}. Model saved.")


        # ====================
    print("Training complete. Running final test evaluation")
    print()
    # Load the best model found during validation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded model from epoch {best_epoch} (best validation C1 F1: {best_val_f1:.4f}).")
    else:
        print("No best model saved. Using the model from the last epoch for testing.")

    # Test the model
    model.eval()
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
            adj = get_adj_unweighted(edges, num_nodes)
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
        adj_est *= max_thres # Rescale the edge weights to the original value range
        # ==========
        # Refine the prediction result
        adj_est = (adj_est+adj_est.T)/2
        # ====================
        # Get the ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_unweighted(edges, num_nodes)
        true_vals = gnd[valid_mask]
        pred_vals = adj_est[valid_mask]

        # ====================
        # Evaluate the quality of current prediction operation

        true_labels = (true_vals >= 1).astype(int)
        pred_labels = (pred_vals >= 1).astype(int)
        precision_per_class, recall_per_class, f1_per_class, _ = \
        precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0, 1], zero_division=0)
        c0precision_list.append(precision_per_class[0])
        c1precision_list.append(precision_per_class[1])
        c0recall_list.append(recall_per_class[0])
        c1recall_list.append(recall_per_class[1])
        c0f1_list.append(f1_per_class[0])
        c1f1_list.append(f1_per_class[1])
        
        # ====================
        # Evaluate the quality of current prediction operation

        c0_prec_mean = np.mean(c0precision_list)
        c0_prec_std = np.std(c0precision_list, ddof=1) if len(c0precision_list) > 1 else 0.0
        c1_prec_mean = np.mean(c1precision_list)
        c1_prec_std = np.std(c1precision_list, ddof=1) if len(c1precision_list) > 1 else 0.0


        c0_recall_mean = np.mean(c0recall_list)
        c0_recall_std = np.std(c0recall_list, ddof=1) if len(c0recall_list) > 1 else 0.0
        c1_recall_mean = np.mean(c1recall_list)
        c1_recall_std = np.std(c1recall_list, ddof=1) if len(c1recall_list) > 1 else 0.0

        c0_f1_mean = np.mean(c0f1_list)
        c0_f1_std = np.std(c0f1_list, ddof=1) if len(c0f1_list) > 1 else 0.0
        c1_f1_mean = np.mean(c1f1_list)
        c1_f1_std = np.std(c1f1_list, ddof=1) if len(c1f1_list) > 1 else 0.0

    # ====================

    print('Test Epoch %d' % epoch)
    print('  C0 Prec: %f (+-%f) C0 Rec: %f (+-%f) C0 F1: %f (+-%f)' %
        (c0_prec_mean, c0_prec_std,
        c0_recall_mean, c0_recall_std,
        c0_f1_mean, c0_f1_std))
    print('  C1 Prec: %f (+-%f) C1 Rec: %f (+-%f) C1 F1: %f (+-%f)' %
        (c1_prec_mean, c1_prec_std,
        c1_recall_mean, c1_recall_std,
        c1_f1_mean, c1_f1_std))
    print()
    # ====================
    print()
    print('Total runtime was: %s seconds' % (time.time() - start_time))

if __name__ == "__main__":
    main()
