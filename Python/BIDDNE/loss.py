import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_DDNE_bipartite_loss(adj_est, gnd_tnr, neigh_tnr, dyn_emb, alpha, beta, weight_class_1=5.0):
    loss_mse = ((adj_est - gnd_tnr)**2)
    weight_mask = (gnd_tnr >= 0.5).float() * weight_class_1 + (gnd_tnr < 0.5).float()
    loss_mse = loss_mse * weight_mask
    loss_mse = torch.sum(loss_mse) / torch.sum(weight_mask)

    loss_reg = torch.sum(torch.norm(dyn_emb, dim=1))
    return loss_mse + beta * loss_reg
