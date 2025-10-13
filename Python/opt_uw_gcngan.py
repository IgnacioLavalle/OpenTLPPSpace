# Demonstration of GCN-GAN

import torch
import torch.optim as optim
from GCN_GAN.modules import *
from GCN_GAN.loss import *
from utils import *
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
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
    parser.add_argument("--trials", type=int, default=400, help="Number of trials")

    return parser.parse_args()


def append_f1_with(c1f1_list, true_labels, pred_labels):
    _, _, f1_per_class, _ = \
        precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0,1], zero_division=0)
    c1f1_list.append(f1_per_class[1])




def objective(trial):
    # ====================
    data_name = 'Recortado677uw'
    num_nodes = 1355 # Number of nodes (Level-1 w/ fixed node set)
    num_snaps = 28 # Number of snapshots
    noise_dim = 32 # Dimension of noise input
    struc_dims = [noise_dim, 32, 16] # Layer configuration of structural encoder
    temp_dims = [num_nodes*struc_dims[-1], 1024] # Layer configuration of temporal encoder
    dec_dims = [temp_dims[-1], num_nodes*num_nodes] # Layer configuration of decoder
    disc_dims = [num_nodes*num_nodes, 512, 256, 64, 1] # Layer configuration of discriminator
    win_size = trial.suggest_categorical("win_size", [2,4,6,8,10]) # Window size of historical snapshots
    alpha = trial.suggest_categorical("alpha", [1, 2, 4, 6, 8, 10, 12]) # Hyper-parameter to adjust the contribution of the MSE loss

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
    valid_mask = np.zeros((1355, 1355), dtype=bool)
    valid_mask[0:137, 137:1355] = True

    #La parte de abajo es el one hot encoding que todavia no se donde meterlo en este codigo
    #node_labels = np.zeros((num_nodes, 2), dtype=np.float32)
    #node_labels[:137, 1] = 1.0
    #node_labels[137:, 0] = 1.0 
    #node_labels_tnr = torch.FloatTensor(node_labels).to(device)

    # ====================
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3, 0.4]) # Dropout rate
    c = c = trial.suggest_categorical("clipping_step", [0.001, 0.005, 0.01, 0.05, 0.1])  # Threshold of the clipping step (for parameters of discriminator)
    num_epochs = 300 # Number of training epochs
    num_val_snaps = 3 # Number of validation snapshots
    num_test_snaps = 3 # Number of test snapshots
    num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots

    # ====================

    # Define the model
    gen_net = GCN_GAN(struc_dims, temp_dims, dec_dims, dropout_rate).to(device) # Generator
    disc_net = DiscNet(disc_dims, dropout_rate).to(device) # Discriminator
    # ==========
    lr_gen = trial.suggest_categorical("lr_g", [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
    lr_disc = trial.suggest_categorical("lr_d", [1e-5, 5e-5, 1e-4, 5e-4])
    weight_decay_val = trial.suggest_categorical("weight_decay", [0.000001, 0.00001, 0.00005, 0.0001, 0.0005, 0.001])
    # Define the optimizer
    gen_opt = optim.RMSprop(gen_net.parameters(), lr=lr_gen, weight_decay=weight_decay_val)
    disc_opt = optim.RMSprop(disc_net.parameters(), lr=lr_disc, weight_decay=weight_decay_val)
    #gen_opt = optim.Adam(gen_net.parameters(), lr=lr_gen, weight_decay=weight_decay_val)
    #disc_opt = optim.Adam(disc_net.parameters(), lr=lr_disc, weight_decay=weight_decay_val)

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
                adj = get_adj_un(edges, num_nodes)
                adj_norm = adj # Normalize the edge weights to [0, 1]
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
            gnd = get_adj_un(edges, num_nodes)
            gnd_norm = gnd # Normalize the edge weights (in ground-truth) to [0, 1]
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
                adj = get_adj_un(edges, num_nodes)
                adj_norm = adj # Normalize the edge weights to [0, 1]
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

            # ====================
            # Get the ground-truth
            edges = edge_seq[tau]
            gnd = get_adj_un(edges, num_nodes)
            # ====================
            # Evaluate the prediction result
            true_vals = gnd[valid_mask]
            pred_vals = adj_est[valid_mask]

            true_labels = (true_vals >= 1).astype(int)
            pred_labels = (pred_vals >= 1).astype(int)
            append_f1_with(c1f1_list, true_labels, pred_labels)
        
        val_c1_f1_mean = np.mean(c1f1_list)
        
    return val_c1_f1_mean



if __name__ == "__main__":
    args = parse_args()
    num_trials = args.trials
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)
    df = study.trials_dataframe()
    df_complete = df[df['state'] == 'COMPLETE']
    columns_of_interest = ["number", "value", "params_dropout_rate", "params_lr_g", "params_lr_d", "params_alpha", "params_weight_decay", "params_clipping_step", "params_win_size"]
    df_filtered = df_complete[columns_of_interest]
    df_filtered.to_csv("optuna_trials_uw_gcngan.csv", index=False)
    print("Se guardaron los trials")
