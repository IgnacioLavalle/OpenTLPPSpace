# Demonstration of DDNE
import time
import torch
import torch.optim as optim
from DDNE.modules import *
from DDNE.loss import *
from utils import *
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of DDNE")
    #adding arguments and their respective default value

    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--epsilon", type=int, default=2, help="Threshold of zero-refining (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--num_val_snaps", type=int, default=3, help="Number of validation snapshots (default: 3)")
    parser.add_argument("--num_test_snaps", type=int, default=3, help="Number of test snapshots (default: 3)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (default: 1e-4)")
    parser.add_argument("--alpha", type=float, default=3.0, help="Alpha value (default: 2.0)")
    parser.add_argument("--beta", type=float, default=0.0, help="Alpha value (default: 0.2)")
    parser.add_argument("--win_size", type=int, default=2, help="Window size of historical snapshots (default: 2)")
    parser.add_argument("--max_thres", type=float, default=2.0, help="Threshold for maximum edge weight (default: 1) (el maximo del grafo es 17500)")

    return parser.parse_args()

def main():
    start_time = time.time()
    args = parse_args()
    misspredicted_1_count = 0
    misspredicted_0_count = 0
    total_elements = 0
    total_edges_greater_equal_1 = 0
    total_edges_lesser_than_1 = 0
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

            # Calculate and store metrics
            RMSE = get_RMSE(adj_est, gnd, num_nodes)
            MAE = get_MAE(adj_est, gnd, num_nodes)            
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
            adj_est, _ = model(adj_list)
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

        print('Test Epoch %d RMSE %f %f MAE %f %f' % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std))
        print()
    # ====================
    """
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
        RMSE = get_RMSE(adj_est, gnd, num_nodes)
        MAE = get_MAE(adj_est, gnd, num_nodes)
        kl = get_EW_KL(adj_est, gnd, num_nodes)

        print(f"Iterative Prediction Test on year {tau - start_test + 1}: RMSE {RMSE}, MAE {MAE}, KL {kl}")
        
        # Classification stats
        total_elements += adj_est.size
        misspredicted_1_matrix = (adj_est >= 1) & (gnd < 1)
        misspredicted_1_count += np.sum(misspredicted_1_matrix)
        misspredicted_0_matrix = (adj_est < 1) & (gnd >= 1)
        misspredicted_0_count += np.sum(misspredicted_0_matrix)
        total_edges_greater_equal_1 += np.sum(gnd >= 1)
        total_edges_lesser_than_1 += (adj_est.size - np.sum(gnd >= 1))
        
        # Update window: we pop the oldest snapshot and them we append the latest prediction
        current_window.pop(0)
        current_window.append(torch.FloatTensor((adj_est / max_thres)).to(device))
    
    #Classification related percentages
    misspredicted_1_percentage = (misspredicted_1_count / total_elements) * 100
    misspredicted_0_percentage = (misspredicted_0_count / total_elements) * 100
    correctly_predicted_percentage = 100 - (misspredicted_1_percentage + misspredicted_0_percentage)
    
    misscaptured_1_edges_percentage = (misspredicted_1_count / total_edges_greater_equal_1) * 100
    correctly_captured_1_edges_percentage = 100 - misscaptured_1_edges_percentage

    misscaptured_0_edges_percentage = (misspredicted_0_count / total_edges_lesser_than_1) * 100
    correctly_captured_0_edges_percentage = 100 - misscaptured_0_edges_percentage

    #Final prints
    print()
    print(f"Classification match percentage: {correctly_predicted_percentage}, Miss-predicted as 0 percentage: {misspredicted_0_percentage}, Miss-predicted as 1 percentage: {misspredicted_1_percentage}")
    print()
    print(f"There were a total of {total_edges_greater_equal_1} edges whose weight was >= 1. {correctly_captured_1_edges_percentage}% were correctly predicted while {misscaptured_1_edges_percentage}% were not")
    print()
    print(f"There were a total of {total_edges_lesser_than_1} edges whose weight was < 1. {correctly_captured_0_edges_percentage}% were correctly predicted while {misscaptured_0_edges_percentage}% were not")
    print()
    print('Total runtime was: %s seconds' % (time.time() - start_time))

if __name__ == "__main__":
    main()
