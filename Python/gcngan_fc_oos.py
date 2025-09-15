# Demonstration of GCN-GAN

import json
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
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, precision_recall_fscore_support, roc_curve,accuracy_score
import warnings


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.enabled = True

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of GCNGAN")
    #adding arguments and their respective default value

    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--epsilon", type=int, default=2, help="Threshold of zero-refining (default: 0.01)")
    parser.add_argument("--num_epochs", type=int, default=260, help="Number of training epochs (default: 100)")
    parser.add_argument("--num_val_snaps", type=int, default=0, help="Number of validation snapshots (default: 3)")
    parser.add_argument("--num_test_snaps", type=int, default=6, help="Number of test snapshots (default: 3)")
    parser.add_argument("--lr_gen", type=float, default=0.00008, help="Learning rate for generator (default: 1e-4)")
    parser.add_argument("--lr_disc", type=float, default=0.0001, help="Learning rate for discriminator (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.00001, help="Weight decay (default: 1e-5)")
    parser.add_argument("--alpha", type=float, default=4.0, help="Alpha value (default: 10.0)")
    parser.add_argument("--win_size", type=int, default=2, help="Window size of historical snapshots (default: 2)")
    parser.add_argument("--max_thres", type=float, default=2.0, help="Threshold for maximum edge weight (default: 1) (el maximo del grafo es 17500)")
    parser.add_argument("--data_name", type=str, default ='Recortado677', help = "Dataset name")
    parser.add_argument("--clipping_step", type=float, default =0.005, help = "Threshold of the clipping step (for parameters of discriminator)")
    parser.add_argument("--patience", type=int, default=10, help="Threshold of early stopping patience (default: 50)")
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

    # ====================
    data_name = args.data_name
    num_nodes = 678 # Number of nodes (Level-1 w/ fixed node set)
    num_snaps = 28 # Number of snapshots
    max_thres = args.max_thres # Threshold for maximum edge weight
    noise_dim = 32 # Dimension of noise input
    struc_dims = [noise_dim, 32, 16] # Layer configuration of structural encoder
    temp_dims = [num_nodes*struc_dims[-1], 1024] # Layer configuration of temporal encoder
    dec_dims = [temp_dims[-1], num_nodes*num_nodes] # Layer configuration of decoder
    disc_dims = [num_nodes*num_nodes, 512, 256, 64, 1] # Layer configuration of discriminator
    win_size = args.win_size # Window size of historical snapshots
    alpha = args.alpha # Hyper-parameter to adjust the contribution of the MSE loss
    save_metrics = args.save_metrics

    # ====================
    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
    valid_mask = np.zeros((678, 678), dtype=bool)
    valid_mask[0:137, 137:678] = True

    #La parte de abajo es el one hot encoding que todavia no se donde meterlo en este codigo
    #node_labels = np.zeros((num_nodes, 2), dtype=np.float32)
    #node_labels[:137, 1] = 1.0
    #node_labels[137:, 0] = 1.0 
    #node_labels_tnr = torch.FloatTensor(node_labels).to(device)

    # ====================
    dropout_rate = args.dropout_rate # Dropout rate
    epsilon = 10 ** (-args.epsilon) # Threshold of zero-refining
    c = args.clipping_step # Threshold of the clipping step (for parameters of discriminator)
    num_epochs = args.num_epochs # Number of training epochs
    num_val_snaps = args.num_val_snaps # Number of validation snapshots
    num_test_snaps = args.num_test_snaps # Number of test snapshots
    num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots

    # ====================

    # Define the model
    gen_net = GCN_GAN(struc_dims, temp_dims, dec_dims, dropout_rate).to(device) # Generator
    disc_net = DiscNet(disc_dims, dropout_rate).to(device) # Discriminator
    # ==========
    lr_gen = args.lr_gen
    lr_disc = args.lr_disc
    weight_decay_val = args.weight_decay
    # Define the optimizer
    gen_opt = optim.RMSprop(gen_net.parameters(), lr=lr_gen, weight_decay=weight_decay_val)
    disc_opt = optim.RMSprop(disc_net.parameters(), lr=lr_disc, weight_decay=weight_decay_val)
    #gen_opt = optim.Adam(gen_net.parameters(), lr=lr_gen, weight_decay=weight_decay_val)
    #disc_opt = optim.Adam(disc_net.parameters(), lr=lr_disc, weight_decay=weight_decay_val)

    best_val_f1 = -1.0 # Or any metric you want to track for 'best' model
    best_epoch = -1
    counter = 0
    patience = args.patience
    best_gen_state = None
    best_disc_state = None



    print(f"data_name: {data_name}, max_thres: {max_thres}, win_size: {win_size}, "
      f"alpha: {alpha}, clipping step: {c}, "
      f"dropout_rate: {dropout_rate}, epsilon: {epsilon}, patience: {patience} "
      f"num_epochs: {num_epochs}, num_val_snaps: {num_val_snaps}, num_test_snaps: {num_test_snaps}, "
      f"num_train_snaps: {num_train_snaps}, lr_gen: {lr_gen}, lr_disc: {lr_disc},  weight_decay_val: {weight_decay_val}")

    print()


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

        """
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

            #Extract as method
            append_classification_metrics_with(c0precision_list, c0recall_list, c0f1_list, c1precision_list, c1recall_list, c1f1_list, true_labels, pred_labels)

            # ====================
        # Evaluate the quality of current prediction operation

        # Classification metrics per class: Precision, Recall, F1
        c0_prec_mean, c0_prec_std, c1_prec_mean, c1_prec_std = mean_and_std_from_classlists(c0precision_list, c1precision_list)

        c0_recall_mean, c0_recall_std, c1_recall_mean, c1_recall_std = mean_and_std_from_classlists(c0recall_list, c1recall_list)

        c0_f1_mean, c0_f1_std, c1_f1_mean, c1_f1_std = mean_and_std_from_classlists(c0f1_list, c1f1_list)


        # ====================

        # Regression metrics: Rmse and Mae
        RMSE_mean = np.mean(RMSE_list)
        RMSE_std = np.std(RMSE_list, ddof=1)
        MAE_mean = np.mean(MAE_list)
        MAE_std = np.std(MAE_list, ddof=1)

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
        
        #EarlyStopping stage
        if c1_f1_mean > best_val_f1:
            best_val_f1 = c1_f1_mean
            best_epoch = epoch
            best_gen_state = gen_net.state_dict()
            best_disc_state = disc_net.state_dict()
            best_gen_opt_state = gen_opt.state_dict()     
            best_disc_opt_state = disc_opt.state_dict()
            counter = 0    

       
        else:
            if epoch >= 200:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered")
                    break
        
        # ====================
        """
    # =========================
    # Test the model
    # ==========
    
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

    print("------- Iterative Prediction with GAN model -------")
    start_test = num_snaps - num_test_snaps

    # Initialize windows with real snapshots
    current_window = []
    for t in range(start_test - win_size, start_test):
        edges = edge_seq[t]
        adj = get_adj_wei(edges, num_nodes, max_thres)
        adj_norm = adj / max_thres
        sup = get_gnn_sup(adj_norm)

        sup_sp = sp.sparse.coo_matrix(sup)
        sup_sp = sparse_to_tuple(sup_sp)
        idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
        vals = torch.FloatTensor(sup_sp[1]).to(device)
        sup_tnr = torch.sparse_coo_tensor(idxs.t(), vals, sup_sp[2], dtype=torch.float32).to(device)
        current_window.append(sup_tnr)

    predictions = []

    for tau in range(start_test, num_snaps):
        gen_net.eval()
        disc_net.eval()

        # generate noise list
        noise_list = []
        for _ in range(win_size):
            noise_feat = gen_noise(num_nodes, noise_dim)
            noise_feat_tnr = torch.FloatTensor(noise_feat).to(device)
            noise_list.append(noise_feat_tnr)

        # === generate prediction ===
        with torch.no_grad():
            adj_est = gen_net(current_window, noise_list)

        adj_est = adj_est.cpu().data.numpy() if torch.cuda.is_available() else adj_est.data.numpy()
        adj_est *= max_thres

        predictions.append(adj_est)

        # === Next iteration set up ===
        adj_norm = adj_est / max_thres
        sup = get_gnn_sup(adj_norm)
        sup_sp = sp.sparse.coo_matrix(sup)
        sup_sp = sparse_to_tuple(sup_sp)
        idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
        vals = torch.FloatTensor(sup_sp[1]).to(device)
        sup_tnr = torch.sparse_coo_tensor(idxs.t(), vals, sup_sp[2], dtype=torch.float32).to(device)

        # Update window
        current_window.pop(0)
        current_window.append(sup_tnr)

        # === ground-truth ===
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)

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

        print(f"Iterative GCNGAN Prediction on snapshot {tau}:")
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

        filename = f"gcngan_fc_curve_metrics_from_year{start_test}.json"
        with open(filename, "w") as f:
            json.dump(metrics_summary, f, indent=2)



    print()
    print('Total runtime was: %s seconds' % (time.time() - start_time))




if __name__ == '__main__':
    main()
