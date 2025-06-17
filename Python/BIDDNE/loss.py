import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_DDNE_bipartite_loss(adj_est, gnd_tnr, emb_u, emb_v, beta=1e-4, weight_class_1=5.0):
    """
    adj_est: [num_u x num_v] matriz reconstruida
    gnd_tnr: [num_u x num_v] matriz ground truth
    emb_u: [num_u x d]
    emb_v: [num_v x d]
    """
    # Weighted MSE
    loss_mse = (adj_est - gnd_tnr) ** 2
    weight_mask = (gnd_tnr >= 1.0).float() * weight_class_1 + (gnd_tnr < 1.0).float()
    loss_mse = loss_mse * weight_mask
    loss_mse = torch.sum(loss_mse) / torch.sum(weight_mask)

    # Regularización L2 sobre los embeddings
    reg_u = torch.sum(torch.norm(emb_u, dim=1))
    reg_v = torch.sum(torch.norm(emb_v, dim=1))
    loss_reg = reg_u + reg_v

    # Total loss
    return loss_mse + beta * loss_reg
