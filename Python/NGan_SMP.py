# Demonstration of NetworkGAN

import torch
import torch.optim as optim
from NetworkGAN.modules import *
from NetworkGAN.loss import *
from utils import *
import time
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of GCNGAN")
    #adding arguments and their respective default value

    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    #parser.add_argument("--epsilon", type=int, default=2, help="Threshold of zero-refining (default: 0.01)")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--num_val_snaps", type=int, default=3, help="Number of validation snapshots (default: 3)")
    parser.add_argument("--num_test_snaps", type=int, default=3, help="Number of test snapshots (default: 3)")
    parser.add_argument("--lr_pg", type=float, default=0.0001, help="Learning rate (default: 1e-4)")
    parser.add_argument("--lr_g", type=float, default=0.0005, help="Learning rate (default: 5e-4)")
    parser.add_argument("--lr_d", type=float, default=0.0005, help="Learning rate (default: 5e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.00001, help="Weight decay (default: 1e-5)")
    parser.add_argument("--win_size", type=int, default=2, help="Window size of historical snapshots (default: 2)")
    parser.add_argument("--max_thres", type=float, default=2.0, help="Threshold for maximum edge weight (default: 1) (el maximo del grafo es 17500)")
    parser.add_argument("--data_name", type=str, default ='SMP22to95', help = "Dataset name")
    parser.add_argument("--gamma", type=float, default=30.0, help="Hyper-parameter to adjust the contribution of the MSE loss (default: 30.0)")


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
    noise_dim = 32 # Dimension of noise input
    struc_dims = [noise_dim, 32, 16] # Layer configuration of structural encoder
    temp_dims = [2*num_nodes*struc_dims[-1], 1024] # Layer configuration of temporal encoder
    dec_dims = [temp_dims[-1]+num_nodes*struc_dims[-1], num_nodes*num_nodes] # Layer configuration of decoder
    disc_dims = [num_nodes*num_nodes, 512, 256, 64, 1] # Layer configuration of discriminator
    win_size = args.win_size # Window size of historical snapshots
    gamma = args.gamma # Hyper-parameter to adjust the contribution of the MSE loss

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True

    # List of TMF features
    # [S, M_list] for each snapshot
    TMF_feat_list = np.load('data/NetworkGAN_TMF_feats_%s.npy' % (data_name), allow_pickle=True)

    # ====================
    dropout_rate = args.dropout_rate # Dropout rate
    #epsilon = 1e-2 # Threshold of zero-refining
    num_pre_epochs = 10 # Number of pre-training epochs
    num_epochs = args.num_epochs # Number of training epochs
    num_val_snaps = args.num_val_snaps # Number of validation snapshots
    num_test_snaps = args.num_test_snaps # Number of test snapshots
    num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots

    # ==========
    none_list = [] # None of TMF features for the first L snapshots
    for t in range(win_size):
        none_list.append(None)
    none_list.extend(TMF_feat_list)
    TMF_feat_list = none_list

    # ====================
    # Define the model
    gen_net = NetworkGAN(struc_dims, temp_dims, dec_dims, dropout_rate).to(device) # Generator
    disc_net = DiscNetDenseF(disc_dims, dropout_rate).to(device) # Discriminator
    # ==========
    # Define the optimizer
    lr_pg = args.lr_pg #pregen lr
    lr_g = args.lr_g #gen lr
    lr_d =args.lr_d #disc lr
    weight_decay=args.weight_decay
    pre_gen_opt = optim.Adam(gen_net.parameters(), lr=lr_pg, weight_decay = weight_decay)
    gen_opt = optim.Adam(gen_net.parameters(), lr=  lr_g, weight_decay = weight_decay)
    disc_opt = optim.Adam(disc_net.parameters(), lr=lr_d, weight_decay = weight_decay)

    print(f"data_name: {data_name}, max_thres: {max_thres}, win_size: {win_size}, "
      f"gamma: {gamma},  "
      f"dropout_rate: {dropout_rate},  "
      f"num_epochs: {num_epochs}, num_val_snaps: {num_val_snaps}, num_test_snaps: {num_test_snaps}, "
      f"num_train_snaps: {num_train_snaps}, pre_gen_lr: {lr_pg}, gen_lr: {lr_g}, disc_lr: {lr_d} , weight_decay_val: {weight_decay}")

    print()

    # ====================
    # Pre-training
    for epoch in range(num_pre_epochs):
        # ====================
        # Pre-train the model
        gen_net.train()
        disc_net.train()
        # ==========
        train_cnt = 0
        gen_loss_list = []
        # ==========
        for tau in range(win_size, num_train_snaps):
            # ====================
            sup_list = [] # List of GNN support (tensor)
            noise_list = [] # List of random noise inputs
            for t in range(tau-win_size, tau):
                # ==========
                edges = edge_seq[t]
                adj = get_adj_wei(edges, num_nodes, max_thres)
                adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
                sup_tnr = torch.FloatTensor(adj_norm).to(device)
                sup_list.append(sup_tnr)
                # ===========
                noise = gen_noise(num_nodes, noise_dim)
                noise_tnr = torch.FloatTensor(noise).to(device)
                noise_list.append(noise_tnr)
            # ==========
            edges = edge_seq[tau]
            gnd = get_adj_wei(edges, num_nodes, max_thres)
            gnd_norm = gnd/max_thres # Normalize the edge weights (in ground-truth) to [0, 1]
            gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
            # ==========
            TMF_feats = TMF_feat_list[tau]
            S = TMF_feats[0]
            S_tnr = torch.FloatTensor(S).to(device)
            M_list_ = TMF_feats[1]
            M_list = []
            for t in range(win_size):
                M = M_list_[t]
                M_tnr = torch.FloatTensor(M).to(device)
                M_list.append(M_tnr)

            # ====================
            for _ in range(1):
                adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
                pre_gen_loss = get_gen_loss_pre(adj_est, gnd_tnr)
                pre_gen_opt.zero_grad()
                pre_gen_loss.backward()
                pre_gen_opt.step()

            # ====================
            gen_loss_list.append(pre_gen_loss.item())
            #print('Pre Loss %f - %d' % (pre_gen_loss.item(), train_cnt))
            train_cnt += 1
            if train_cnt % 100 == 0:
                print('-Train %d / %d' % (train_cnt, num_train_snaps))
        gen_loss_mean = np.mean(gen_loss_list)
        print('#%d Pre-Train G-Loss %f' % (epoch, gen_loss_mean))

        # ====================
        # Validate the model
        gen_net.eval()
        disc_net.eval()
        # ==========
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
            sup_list = [] # List of GNN support (tensor)
            noise_list = [] # List of random noise inputs
            for t in range(tau-win_size, tau):
                # ==========
                edges = edge_seq[t]
                adj = get_adj_wei(edges, num_nodes, max_thres)
                adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
                sup_tnr = torch.FloatTensor(adj_norm).to(device)
                sup_list.append(sup_tnr)
                # ===========
                noise = gen_noise(num_nodes, noise_dim)
                noise_tnr = torch.FloatTensor(noise).to(device)
                noise_list.append(noise_tnr)
            # ==========
            TMF_feats = TMF_feat_list[tau]
            S = TMF_feats[0]
            S_tnr = torch.FloatTensor(S).to(device)
            M_list_ = TMF_feats[1]
            M_list = []
            for t in range(win_size):
                M = M_list_[t]
                M_tnr = torch.FloatTensor(M).to(device)
                M_list.append(M_tnr)

            # ====================
            # Get the prediction result
            adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
            if torch.cuda.is_available():
                adj_est = adj_est.cpu().data.numpy()
            else:
                adj_est = adj_est.data.numpy()
            adj_est *= max_thres # Rescale the edge weights to the original value range
            # ==========
            # ====================
            # Get the ground-truth
            edges = edge_seq[tau]
            gnd = get_adj_wei(edges, num_nodes, max_thres)
            # ====================
            # Evaluate the prediction result
            true_vals = gnd[valid_mask]
            pred_vals = adj_est[valid_mask]

            true_labels = (true_vals >= 1).astype(int)
            pred_labels = (pred_vals >= 1).astype(int)

            #Errors
            abs_errors = np.abs(pred_vals - true_vals)
            sq_errors = (pred_vals - true_vals) ** 2

            #MAE and std
            filtered_mae = np.mean(abs_errors)

            #RMSE and std
            filtered_rmse = np.sqrt(np.mean(sq_errors))

            RMSE = filtered_rmse
            MAE = filtered_mae
            # ====================
        RMSE_mean = np.mean(RMSE_list)
        RMSE_std = np.std(RMSE_list, ddof=1)
        MAE_mean = np.mean(MAE_list)
        MAE_std = np.std(MAE_list, ddof=1)
        append_classification_metrics_with(c0precision_list, c0recall_list, c0f1_list, c1precision_list, c1recall_list, c1f1_list, true_labels, pred_labels)

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
        
    # ====================
    # Test the model
    gen_net.eval()
    disc_net.eval()
    # ==========
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
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
            sup_tnr = torch.FloatTensor(adj_norm).to(device)
            sup_list.append(sup_tnr)
            # ===========
            noise = gen_noise(num_nodes, noise_dim)
            noise_tnr = torch.FloatTensor(noise).to(device)
            noise_list.append(noise_tnr)
        # ==========
        TMF_feats = TMF_feat_list[tau]
        S = TMF_feats[0]
        S_tnr = torch.FloatTensor(S).to(device)
        M_list_ = TMF_feats[1]
        M_list = []
        for t in range(win_size):
            M = M_list_[t]
            M_tnr = torch.FloatTensor(M).to(device)
            M_list.append(M_tnr)

        # ====================
        # Get the prediction result
        adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        adj_est *= max_thres  # Rescale the edge weights to the original value range
        # ==========
        # ====================
        # Get the ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ====================
        # Evaluate the prediction result
        true_vals = gnd[valid_mask]
        pred_vals = adj_est[valid_mask]

        true_labels = (true_vals >= 1).astype(int)
        pred_labels = (pred_vals >= 1).astype(int)

        #Errors
        abs_errors = np.abs(pred_vals - true_vals)
        sq_errors = (pred_vals - true_vals) ** 2

        #MAE and std
        filtered_mae = np.mean(abs_errors)

        #RMSE and std
        filtered_rmse = np.sqrt(np.mean(sq_errors))

        RMSE = filtered_rmse
        MAE = filtered_mae

        # ==========
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
        # ====================
    RMSE_mean = np.mean(RMSE_list)
    RMSE_std = np.std(RMSE_list, ddof=1)
    MAE_mean = np.mean(MAE_list)
    MAE_std = np.std(MAE_list, ddof=1)

    append_classification_metrics_with(c0precision_list, c0recall_list, c0f1_list, c1precision_list, c1recall_list, c1f1_list, true_labels, pred_labels)

    # Classification metrics per class: Precision, Recall, F1
    c0_prec_mean, c0_prec_std, c1_prec_mean, c1_prec_std = mean_and_std_from_classlists(c0precision_list, c1precision_list)

    c0_recall_mean, c0_recall_std, c1_recall_mean, c1_recall_std = mean_and_std_from_classlists(c0recall_list, c1recall_list)

    c0_f1_mean, c0_f1_std, c1_f1_mean, c1_f1_std = mean_and_std_from_classlists(c0f1_list, c1f1_list)

    print('Test Pre Epoch %d RMSE %f %f MAE %f %f'
        % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std))
    print('  C0 Prec: %f (+-%f) C0 Rec: %f (+-%f) C0 F1: %f (+-%f)' %
        (c0_prec_mean, c0_prec_std,
        c0_recall_mean, c0_recall_std,
        c0_f1_mean, c0_f1_std))
    print('  C1 Prec: %f (+-%f) C1 Rec: %f (+-%f) C1 F1: %f (+-%f)' %
        (c1_prec_mean, c1_prec_std,
        c1_recall_mean, c1_recall_std,
        c1_f1_mean, c1_f1_std))


    # ====================
    # Formal optimization
    for epoch in range(num_epochs):
        # ====================
        # Train the model
        gen_net.train()
        disc_net.train()
        # ==========
        train_cnt = 0
        disc_loss_list = []
        gen_loss_list = []
        for tau in range(win_size, num_train_snaps):
            # ====================
            sup_list = []  # List of GNN support (tensor)
            noise_list = [] # List of random noise inputs
            for t in range(tau-win_size, tau):
                # ==========
                edges = edge_seq[t]
                adj = get_adj_wei(edges, num_nodes, max_thres)
                adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
                sup_tnr = torch.FloatTensor(adj_norm).to(device)
                sup_list.append(sup_tnr)
                # ===========
                noise = gen_noise(num_nodes, noise_dim)
                noise_tnr = torch.FloatTensor(noise).to(device)
                noise_list.append(noise_tnr)
            # ==========
            edges = edge_seq[tau]
            gnd = get_adj_wei(edges, num_nodes, max_thres)
            gnd_norm = gnd/max_thres # Normalize the edge weights (in ground-truth) to [0, 1]
            gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
            # ==========
            TMF_feats = TMF_feat_list[tau]
            S = TMF_feats[0]
            S_tnr = torch.FloatTensor(S).to(device)
            M_list_ = TMF_feats[1]
            M_list = []
            for t in range(win_size):
                M = M_list_[t]
                M_tnr = torch.FloatTensor(M).to(device)
                M_list.append(M_tnr)

            # ====================
            for _ in range(1):
                # ==========
                # Train the discriminator
                noise = gen_noise(num_nodes, noise_dim)
                noise_tnr = torch.FloatTensor(noise).to(device)
                adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
                disc_real, disc_fake = disc_net(gnd_tnr, adj_est, num_nodes)
                disc_loss = get_disc_loss(disc_real, disc_fake) # Loss of the discriminative network
                disc_opt.zero_grad()
                disc_loss.backward()
                disc_opt.step()
                # ==========
                # Train the generator
                noise = gen_noise(num_nodes, noise_dim)
                adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
                _, disc_fake = disc_net(gnd_tnr, adj_est, num_nodes)
                gen_loss = get_gen_loss(adj_est, gnd_tnr, disc_fake, gamma)
                gen_opt.zero_grad()
                gen_loss.backward()
                gen_opt.step()

            # ====================
            gen_loss_list.append(gen_loss.item())
            disc_loss_list.append(disc_loss.item())
            #print('Pre G-Loss %f D-Loss %f - %d' % (gen_loss.item(), disc_loss.item(), train_cnt))
            train_cnt += 1
            if train_cnt % 100 == 0:
                print('-Train %d / %d' % (train_cnt, num_train_snaps))
        gen_loss_mean = np.mean(gen_loss_list)
        disc_loss_mean = np.mean(disc_loss_list)
        print('#%d Train G-Loss %f D-Loss %f' % (epoch, gen_loss_mean, disc_loss_mean))

        # ====================
        # Validate the model
        gen_net.eval()
        disc_net.eval()
        # ==========
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
            sup_list = [] # List of GNN support (tensor)
            noise_list = [] # List of random noise inputs
            for t in range(tau-win_size, tau):
                # ==========
                edges = edge_seq[t]
                adj = get_adj_wei(edges, num_nodes, max_thres)
                adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
                sup_tnr = torch.FloatTensor(adj_norm).to(device)
                sup_list.append(sup_tnr)
                # ===========
                noise = gen_noise(num_nodes, noise_dim)
                noise_tnr = torch.FloatTensor(noise).to(device)
                noise_list.append(noise_tnr)
            # ==========
            TMF_feats = TMF_feat_list[tau]
            S = TMF_feats[0]
            S_tnr = torch.FloatTensor(S).to(device)
            M_list_ = TMF_feats[1]
            M_list = []
            for t in range(win_size):
                M = M_list_[t]
                M_tnr = torch.FloatTensor(M).to(device)
                M_list.append(M_tnr)

            # ====================
            # Get the prediction result
            adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
            if torch.cuda.is_available():
                adj_est = adj_est.cpu().data.numpy()
            else:
                adj_est = adj_est.data.numpy()
            adj_est *= max_thres # Rescale the edge weights to the original value range
            # ==========
            # ====================
            # Get the ground-truth
            edges = edge_seq[tau]
            gnd = get_adj_wei(edges, num_nodes, max_thres)
            # ====================
            # Evaluate the prediction result
            true_vals = gnd[valid_mask]
            pred_vals = adj_est[valid_mask]

            true_labels = (true_vals >= 1).astype(int)
            pred_labels = (pred_vals >= 1).astype(int)

            #Errors
            abs_errors = np.abs(pred_vals - true_vals)
            sq_errors = (pred_vals - true_vals) ** 2

            #MAE and std
            filtered_mae = np.mean(abs_errors)

            #RMSE and std
            filtered_rmse = np.sqrt(np.mean(sq_errors))

            RMSE = filtered_rmse
            MAE = filtered_mae

            # ==========
            RMSE_list.append(RMSE)
            MAE_list.append(MAE)
            # ====================
        RMSE_mean = np.mean(RMSE_list)
        RMSE_std = np.std(RMSE_list, ddof=1)
        MAE_mean = np.mean(MAE_list)
        MAE_std = np.std(MAE_list, ddof=1)
        append_classification_metrics_with(c0precision_list, c0recall_list, c0f1_list, c1precision_list, c1recall_list, c1f1_list, true_labels, pred_labels)

        # Classification metrics per class: Precision, Recall, F1
        c0_prec_mean, c0_prec_std, c1_prec_mean, c1_prec_std = mean_and_std_from_classlists(c0precision_list, c1precision_list)

        c0_recall_mean, c0_recall_std, c1_recall_mean, c1_recall_std = mean_and_std_from_classlists(c0recall_list, c1recall_list)

        c0_f1_mean, c0_f1_std, c1_f1_mean, c1_f1_std = mean_and_std_from_classlists(c0f1_list, c1f1_list)

        print('Val Epoch %d RMSE %f %f MAE %f %f'
            % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std))
        print('  C0 Prec: %f (+-%f) C0 Rec: %f (+-%f) C0 F1: %f (+-%f)' %
            (c0_prec_mean, c0_prec_std,
            c0_recall_mean, c0_recall_std,
            c0_f1_mean, c0_f1_std))
        print('  C1 Prec: %f (+-%f) C1 Rec: %f (+-%f) C1 F1: %f (+-%f)' %
            (c1_prec_mean, c1_prec_std,
            c1_recall_mean, c1_recall_std,
            c1_f1_mean, c1_f1_std))

    # ====================
    # Test the model
    gen_net.eval()
    disc_net.eval()
    # ==========
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
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres  # Normalize the edge weights to [0, 1]
            sup_tnr = torch.FloatTensor(adj_norm).to(device)
            sup_list.append(sup_tnr)
            # ===========
            noise = gen_noise(num_nodes, noise_dim)
            noise_tnr = torch.FloatTensor(noise).to(device)
            noise_list.append(noise_tnr)
        # ==========
        TMF_feats = TMF_feat_list[tau]
        S = TMF_feats[0]
        S_tnr = torch.FloatTensor(S).to(device)
        M_list_ = TMF_feats[1]
        M_list = []
        for t in range(win_size):
            M = M_list_[t]
            M_tnr = torch.FloatTensor(M).to(device)
            M_list.append(M_tnr)

        # ====================
        # Get the prediction result
        adj_est = gen_net(sup_list, noise_list, S_tnr, M_list)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        adj_est *= max_thres  # Rescale the edge weights to the original value range
        # ==========
        # ====================
        # Get the ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ====================
        # Evaluate the prediction result
        true_vals = gnd[valid_mask]
        pred_vals = adj_est[valid_mask]

        true_labels = (true_vals >= 1).astype(int)
        pred_labels = (pred_vals >= 1).astype(int)

        #Errors
        abs_errors = np.abs(pred_vals - true_vals)
        sq_errors = (pred_vals - true_vals) ** 2

        #MAE and std
        filtered_mae = np.mean(abs_errors)

        #RMSE and std
        filtered_rmse = np.sqrt(np.mean(sq_errors))

        RMSE = filtered_rmse
        MAE = filtered_mae

        # ==========
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)

        # ====================
    RMSE_mean = np.mean(RMSE_list)
    RMSE_std = np.std(RMSE_list, ddof=1)
    MAE_mean = np.mean(MAE_list)
    MAE_std = np.std(MAE_list, ddof=1)

    append_classification_metrics_with(c0precision_list, c0recall_list, c0f1_list, c1precision_list, c1recall_list, c1f1_list, true_labels, pred_labels)

    # Classification metrics per class: Precision, Recall, F1
    c0_prec_mean, c0_prec_std, c1_prec_mean, c1_prec_std = mean_and_std_from_classlists(c0precision_list, c1precision_list)

    c0_recall_mean, c0_recall_std, c1_recall_mean, c1_recall_std = mean_and_std_from_classlists(c0recall_list, c1recall_list)

    c0_f1_mean, c0_f1_std, c1_f1_mean, c1_f1_std = mean_and_std_from_classlists(c0f1_list, c1f1_list)

    print('Test Epoch %d RMSE %f %f MAE %f %f'
        % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std))
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
