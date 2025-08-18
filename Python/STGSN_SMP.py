# Demonstration of STGSN

import argparse
import time
import torch
import torch.optim as optim
from STGSN.modules import *
from STGSN.loss import *
from utils import *
from sklearn.metrics import precision_recall_fscore_support

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstration of STSGN")
    #adding arguments and their respective default value

    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--epsilon", type=int, default=2, help="Threshold of zero-refining (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--num_val_snaps", type=int, default=3, help="Number of validation snapshots (default: 3)")
    parser.add_argument("--num_test_snaps", type=int, default=3, help="Number of test snapshots (default: 3)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (default: 1e-4)")
    parser.add_argument("--theta", type=float, default=0.1, help="theta value (default: 0.1)")
    parser.add_argument("--win_size", type=int, default=2, help="Window size of historical snapshots (default: 2)")
    parser.add_argument("--max_thres", type=float, default=1.0, help="Threshold for maximum edge weight (default: 1) (el maximo del grafo es 17500)")
    parser.add_argument("--data_name", type=str, default ='SMP22to95', help = "Dataset name")
    parser.add_argument("--feat_name", type=str, default ='SMP22to95_oh', help = "Dataset features name")
    parser.add_argument("--feat_dim", type=int, default=32, help="dimensionality of feautre input (default: 32)")
    parser.add_argument("--enc_dim", type=int, default=32, help="dimensionality of enconder (default: 32)")
    parser.add_argument("--diff_dim", type=bool, default=False, help="Decides wether encoder dimensions are all the same or not (Default: False)")



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
    diff_dim = args.diff_dim
    f_layer = s_layer = t_layer = args.enc_dim
    #If you choose enconder dimension to be the same you have [feat_dim, f_layer,f_layer,f_layer]
    #So if instead you choose encoder dimensions to be different, you have [feat_dim, f_layer,f_layer * 2,f_layer * 4]
    if diff_dim:
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
                    sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
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
                col_sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, col_sup_sp[2]).float().to(device)
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
        # Validate the model
        model.eval()
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
            col_net = np.zeros((num_nodes, num_nodes))
            coef_sum = 0.0
            for t in range(tau-win_size, tau):
                # ==========
                edges = edge_seq[t]
                adj = get_adj_wei(edges, num_nodes, max_thres)
                adj_norm = adj / max_thres
                sup = get_gnn_sup_d(adj_norm)
                sup_sp = sp.sparse.coo_matrix(sup)
                sup_sp = sparse_to_tuple(sup_sp)
                idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
                vals = torch.FloatTensor(sup_sp[1]).to(device)
                sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
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
            col_sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, col_sup_sp[2]).float().to(device)
            # ==========
            # Get the prediction result
            adj_est = model(sup_list, feat_list, col_sup_tnr, feat_tnr, num_nodes)
            if torch.cuda.is_available():
                adj_est = adj_est.cpu().data.numpy()
            else:
                adj_est = adj_est.data.numpy()
            adj_est *= max_thres  # Rescale the edge weights to the original value range
            # ==========
            # Refine the prediction result
            adj_est = (adj_est+adj_est.T)/2

            # ====================
            # Get the ground-truth
            edges = edge_seq[tau]
            gnd = get_adj_wei(edges, num_nodes, max_thres)
            # ====================
            # ====================
            # Evaluate the prediction result
            true_vals = gnd[valid_mask]
            pred_vals = adj_est[valid_mask]

            true_labels = (true_vals >= 1).astype(int)
            pred_labels = (pred_vals >= 1).astype(int)

            mask_class_1 = (true_labels == 1)
            mask_class_0 = (true_labels == 0)


            #Errors
            abs_errors = np.abs(pred_vals - true_vals)
            sq_errors = (pred_vals - true_vals) ** 2

            mae_class_1 = np.mean(abs_errors[mask_class_1])
            rmse_class_1 = np.sqrt(np.mean(sq_errors[mask_class_1]))
            mae_class_0 = np.mean(abs_errors[mask_class_0])
            rmse_class_0 = np.sqrt(np.mean(sq_errors[mask_class_0]))


            #MAE and std
            MAE = np.mean(abs_errors)

            #RMSE and std
            RMSE = np.sqrt(np.mean(sq_errors))

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
        
        if c1_f1_mean > best_val_f1:
                best_val_f1 = c1_f1_mean
                best_epoch = epoch
    # ====================
    # Test the model
    model.eval()
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
            sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
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
        col_sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, col_sup_sp[2]).float().to(device)
        # ==========
        # Get the prediction result
        adj_est = model(sup_list, feat_list, col_sup_tnr, feat_tnr, num_nodes)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        adj_est *= max_thres  # Rescale the edge weights to the original value range
        # ==========
        # Refine the prediction result
        adj_est = (adj_est+adj_est.T)/2


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

        mask_class_1 = (true_labels == 1)
        mask_class_0 = (true_labels == 0)


        #Errors
        abs_errors = np.abs(pred_vals - true_vals)
        sq_errors = (pred_vals - true_vals) ** 2

        mae_class_1 = np.mean(abs_errors[mask_class_1])
        rmse_class_1 = np.sqrt(np.mean(sq_errors[mask_class_1]))
        mae_class_0 = np.mean(abs_errors[mask_class_0])
        rmse_class_0 = np.sqrt(np.mean(sq_errors[mask_class_0]))


        #MAE and std
        MAE = np.mean(abs_errors)

        #RMSE and std
        RMSE = np.sqrt(np.mean(sq_errors))

        # ==========
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)

        #Extract as method
        append_classification_metrics_with(c0precision_list, c0recall_list, c0f1_list, c1precision_list, c1recall_list, c1f1_list, true_labels, pred_labels)

        print('Test snapshot %d metrics per snapshot:'
            % (tau))
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
        print(f"  MAE on class 1 ground truth: {mae_class_1} , RMSE on class 1: {rmse_class_1}")
        print(f"  MAE on class 0 ground truth: {mae_class_0} , RMSE on class 0: {rmse_class_0}")
        print()

    # Evaluate the quality of current prediction operation

    # Classification metrics per class: Precision, Recall, F1
    c0_prec_mean, c0_prec_std, c1_prec_mean, c1_prec_std = mean_and_std_from_classlists(c0precision_list, c1precision_list)

    c0_recall_mean, c0_recall_std, c1_recall_mean, c1_recall_std = mean_and_std_from_classlists(c0recall_list, c1recall_list)

    c0_f1_mean, c0_f1_std, c1_f1_mean, c1_f1_std = mean_and_std_from_classlists(c0f1_list, c1f1_list)


    # ====================
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
