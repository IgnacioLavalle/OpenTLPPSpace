import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_DDNE_bipartite_loss(adj_est, gnd, neigh, emb_U, emb_V, alpha, beta):
    '''
    DDNE loss adaptada para grafos bipartitos U-V.

    :param adj_est: matriz estimada de adyacencia (|U| x |V|)
    :param gnd: matriz ground-truth (|U| x |V|)
    :param neigh: matriz de frecuencias o pesos de conexiones (|U| x |V|) — opcional
    :param emb_U: embedding nodos U (|U| x d)
    :param emb_V: embedding nodos V (|V| x d)
    :param alpha: peso para error en aristas existentes
    :param beta: peso para regularización
    '''
    # Máscara P que da más peso a las aristas existentes
    P = torch.ones_like(gnd)
    P = torch.where(gnd != 0, alpha * P, P)

    # Pérdida de reconstrucción (error cuadrático ponderado)
    recon_loss = torch.norm((adj_est - gnd) * P, p='fro')**2

    # Regularización de suavidad temporal basada en distancias entre embeddings con peso neigh
    if neigh is not None:
        dist_matrix = torch.cdist(emb_U, emb_V, p=2)  # (|U| x |V|)
        reg_loss = torch.sum(neigh * (dist_matrix ** 2))
    else:
        reg_loss = torch.norm(emb_U, p='fro')**2 + torch.norm(emb_V, p='fro')**2

    loss = recon_loss + (beta / 2) * reg_loss
    return loss
