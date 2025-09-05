# Demonstration of GCN-GAN

import torch
import torch.optim as optim
from GCN_GAN.modules import *
from GCN_GAN.loss import *
from utils import *
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import warnings
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.enabled = True

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of GCNGAN")
    #adding arguments and their respective default value
    parser.add_argument("--trials", type=int, default=200, help="Number of trials")

    return parser.parse_args()

def append_f1_with(c1f1_list, true_labels, pred_labels):
    _, _, f1_per_class, _ = \
        precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0,1], zero_division=0)
    c1f1_list.append(f1_per_class[1])



def objective(trial):

    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step = 0.1) # Dropout rate
    win_size = trial.suggest_categorical("win_size", [1,2,4]) # Window size of historical snapshots
    lr_gen = trial.suggest_categorical("lrgen", [0.00001, 0.00005, 0.00008, 0.0001, 0.0005, 0.001])
    weight_decay_val = trial.suggest_categorical("weight_decay", [0.000005,0.00001,0.00005, 0.0001, 0.001])
    c = trial.suggest_categorical("cstep", [0.001,0.005,0.01]) # Threshold of the clipping step (for parameters of discriminator)
    alpha = trial.suggest_categorical("alpha", [1.0, 3.0, 4.0, 7.0, 9.0, 12.0]) # Hyper-parameter to adjust the contribution of the MSE loss
    

    # ====================
    #lr_disc = trial.suggest_categorical("lrdisc", [0.00001, 0.00005, 0.00008, 0.0001, 0.0005, 0.001])
    lr_disc = 0.0001
    data_name = 'Recortado677'
    num_nodes = 1355 # Number of nodes (Level-1 w/ fixed node set)
    num_snaps = 28 # Number of snapshots
    max_thres = 1.5 # Threshold for maximum edge weight
    noise_dim = 32 # Dimension of noise input
    struc_dims = [noise_dim, 32, 16] # Layer configuration of structural encoder
    temp_dims = [num_nodes*struc_dims[-1], 1024] # Layer configuration of temporal encoder
    dec_dims = [temp_dims[-1], num_nodes*num_nodes] # Layer configuration of decoder
    disc_dims = [num_nodes*num_nodes, 512, 256, 64, 1] # Layer configuration of discriminator
    
    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True

    # ====================
    num_epochs = 260 # Number of training epochs
    num_val_snaps = 3 # Number of validation snapshots
    num_test_snaps = 3 # Number of test snapshots
    num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots

    # ====================

    # Define the model
    gen_net = GCN_GAN(struc_dims, temp_dims, dec_dims, dropout_rate).to(device) # Generator
    disc_net = DiscNet(disc_dims, dropout_rate).to(device) # Discriminator
    # ==========
    # Define the optimizer
    gen_opt = optim.RMSprop(gen_net.parameters(), lr=lr_gen, weight_decay=weight_decay_val)
    disc_opt = optim.RMSprop(disc_net.parameters(), lr=lr_disc, weight_decay=weight_decay_val)
    #gen_opt = optim.Adam(gen_net.parameters(), lr=1e-4, weight_decay=0)
    #disc_opt = optim.Adam(disc_net.parameters(), lr=1e-4, weight_decay=0)

    # ====================
    for epoch in range(num_epochs):
        # ====================
        # Training the model
        gen_net.train()
        disc_net.train()
        # ==========
        train_cnt = 0
        disc_loss_list = []
        gen_loss_list = []
        for tau in range(win_size, num_train_snaps):
            # ====================
            sup_list = [] # List of GNN support (tensor)
            noise_list = [] # List of noise input
            for t in range(tau-win_size, tau):
                # ==========
                edges = edge_seq[t]
                adj = get_adj_wei(edges, num_nodes, max_thres)
                adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
                # ==========
                # Transfer the GNN support to a sparse tensor
                sup = get_gnn_sup(adj_norm)
                sup_sp = sp.sparse.coo_matrix(sup)
                sup_sp = sparse_to_tuple(sup_sp)
                #idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
                idxs = torch.LongTensor(sup_sp[0]).to(device)
                vals = torch.FloatTensor(sup_sp[1]).to(device)
                #sup_tnr = torch.sparse_coo_tensor(idxs.t(), vals, sup_sp[2]).float().to(device)
                sup_tnr = torch.sparse_coo_tensor(idxs.t(), vals, sup_sp[2], dtype=torch.float32).to(device)
                sup_list.append(sup_tnr)
                # =========
                # Generate random noise
                noise_feat = gen_noise(num_nodes, noise_dim)
                noise_feat_tnr = torch.FloatTensor(noise_feat).to(device)
                noise_list.append(noise_feat_tnr)
            # ==========
            edges = edge_seq[tau]
            gnd = get_adj_wei(edges, num_nodes, max_thres)
            gnd_norm = gnd/max_thres # Normalize the edge weights (in ground-truth) to [0, 1]
            gnd_tnr = torch.FloatTensor(gnd_norm).to(device)

            for _ in range(1):
                # ====================
                # Train the discriminator
                #adj_est = gen_net(sup_list, noise_list)
                adj_est = gen_net(sup_list, noise_list).detach()
                disc_real, disc_fake = disc_net(gnd_tnr, adj_est, num_nodes)
                disc_loss = get_disc_loss(disc_real, disc_fake) # Loss of the discriminator
                disc_opt.zero_grad()
                #Al ejecutar la siguiente linea muere
                disc_loss.backward()
                #Aca ya murió
                disc_opt.step()
                # ===========
                # Clip parameters of discriminator
                for param in disc_net.parameters():
                    param.data.clamp_(-c, c)
                # ==========
                # Train the generative network
                adj_est = gen_net(sup_list, noise_list)
                _, disc_fake = disc_net(gnd_tnr, adj_est, num_nodes)
                gen_loss = get_gen_loss(adj_est, gnd_tnr, disc_fake, alpha) # Loss of the generative network
                gen_opt.zero_grad()
                gen_loss.backward()
                gen_opt.step()

            # ====================
            gen_loss_list.append(gen_loss.item())
            disc_loss_list.append(disc_loss.item())
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
        c1f1_list = []

        for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
            # ====================
            sup_list = [] # List of GNN support (tensor)
            noise_list = [] # List of noise input
            for t in range(tau-win_size, tau):
                # ==========
                edges = edge_seq[t]
                adj = get_adj_wei(edges, num_nodes, max_thres)
                adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
                sup = get_gnn_sup(adj_norm)
                # ==========
                # Transfer the GNN support to a sparse tensor
                sup_sp = sp.sparse.coo_matrix(sup)
                sup_sp = sparse_to_tuple(sup_sp)
                idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
                vals = torch.FloatTensor(sup_sp[1]).to(device)
                #sup_tnr = torch.sparse_coo_tensor(idxs.t(), vals, sup_sp[2]).float().to(device)
                sup_tnr = torch.sparse_coo_tensor(idxs.t(), vals, sup_sp[2], dtype=torch.float32).to(device)
                sup_list.append(sup_tnr)
                # ==========
                # Generate random noise
                noise_feat = gen_noise(num_nodes, noise_dim)
                noise_feat_tnr = torch.FloatTensor(noise_feat).to(device)
                noise_list.append(noise_feat_tnr)
            # ====================
            # Get the prediction result
            adj_est = gen_net(sup_list, noise_list)
            if torch.cuda.is_available():
                adj_est = adj_est.cpu().data.numpy()
            else:
                adj_est = adj_est.data.numpy()
            adj_est *= max_thres  # Rescale the edge weights to the original value range

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

            append_f1_with(c1f1_list, true_labels, pred_labels)
        
        val_c1_f1_mean = np.mean(c1f1_list)
        
    return val_c1_f1_mean

if __name__ == '__main__':
    args = parse_args()
    num_trials = args.trials
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)
    df = study.trials_dataframe()
    df_complete = df[df['state'] == 'COMPLETE']
    columns_of_interest = ["number", "value", "params_dropout_rate", "params_lrgen", "params_lrdisc", "params_alpha", "params_cstep", "params_win_size"]
    df_filtered = df_complete[columns_of_interest]
    df_filtered.to_csv("optuna_trials_gcngan.csv", index=False)
    print("Se guardaron los trials")
