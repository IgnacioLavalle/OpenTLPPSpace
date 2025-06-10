import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_DDNE_loss(adj_est, gnd, neigh, emb, alpha, beta):
    '''
    Function to define the loss of DDNE
    :param adj_est: prediction result (the estimated adjacency matrix)
    :param gnd: ground-truth (adjacency matrix of the next snapshot)
    :param neigh: connection frequency matrix
    :param emb: learned temporal embedding
    :param alpha, beta: hyper-parameters
    :return: loss of DDNE
    '''
    valid_mask_np = np.zeros((1355, 1355), dtype=bool)
    valid_mask_np[0:137, 137:1355] = True

    # ====================
    P = torch.ones_like(gnd)
    P_alpha = alpha*P
    P = torch.where(gnd==0, P, P_alpha)
    #loss = torch.norm(torch.mul((adj_est - gnd), P), p='fro')**2
    
    ### Seccion nueva, para volver a lo viejo simplemente hay que borrar de aca hasta el ###
    weighted_diff = torch.mul((adj_est - gnd), P)
    valid_mask = torch.from_numpy(valid_mask_np).to(gnd.device)
    loss_term1_elements = weighted_diff[valid_mask]
    loss = torch.norm(loss_term1_elements, p='fro')**2
    ###


    deg = torch.diag(torch.sum(neigh, dim=0))
    lap = deg - neigh # Laplacian matrix of connection frequency
    loss += (beta/2)*torch.trace(torch.mm(emb.t(), torch.mm(lap, emb)))

    return loss