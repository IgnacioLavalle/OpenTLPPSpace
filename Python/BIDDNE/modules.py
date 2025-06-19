import torch
import torch.nn as nn

class DDNE(nn.Module):
    '''
    DDNE for bipartite graphs).
    '''
    def __init__(self, enc_dims, dec_dims, dropout_rate, num_u=137, num_v=1218):
        super(DDNE, self).__init__()
        self.enc_dims = enc_dims
        self.dec_dims = dec_dims
        self.dropout_rate = dropout_rate
        self.num_u = num_u
        self.num_v = num_v

        self.encoder = DDNE_Enc(enc_dims, dropout_rate, num_u, num_v)
        self.decoder = DDNE_Dec()

    def forward(self, adj_list):
        emb_u, emb_v = self.encoder(adj_list)
        adj_est = self.decoder(emb_u, emb_v)
        return adj_est, (emb_u, emb_v)

class DDNE_Enc(nn.Module):
    '''
    BIDDNE Encoder.
    '''
    def __init__(self, enc_dims, dropout_rate, num_u=137, num_v=1218):
        super(DDNE_Enc, self).__init__()
        self.enc_dims = enc_dims
        self.dropout_rate = dropout_rate
        self.num_layers = len(enc_dims) - 1

        # Separated GRUs for nodes u and v
        self.for_RNNs_u = nn.ModuleList()
        self.rev_RNNs_u = nn.ModuleList()
        self.for_RNNs_v = nn.ModuleList()
        self.rev_RNNs_v = nn.ModuleList()

        for i in range(self.num_layers):
            self.for_RNNs_u.append(
                nn.GRU(input_size=enc_dims[i], hidden_size=enc_dims[i + 1], batch_first=False)
            )
            self.rev_RNNs_u.append(
                nn.GRU(input_size=enc_dims[i], hidden_size=enc_dims[i + 1], batch_first=False)
            )
            self.for_RNNs_v.append(
                nn.GRU(input_size=num_u if i == 0 else enc_dims[i], hidden_size=enc_dims[i + 1], batch_first=False)
            )
            self.rev_RNNs_v.append(
                nn.GRU(input_size=num_u if i == 0 else enc_dims[i], hidden_size=enc_dims[i + 1], batch_first=False)
            )

    def forward(self, adj_list):
        win_size = len(adj_list)

        # u_seq: [T, num_u, num_v]
        u_seq = torch.stack(adj_list, dim=0)  # [T, 137, 1218]
        v_seq = torch.stack([A.transpose(0, 1) for A in adj_list], dim=0)  # [T, 1218, 137]

        # u nodes processing
        for_in_u = u_seq
        rev_in_u = torch.flip(u_seq, dims=[0])
        for l in range(self.num_layers):
            for_out_u, _ = self.for_RNNs_u[l](for_in_u)
            rev_out_u, _ = self.rev_RNNs_u[l](rev_in_u)
            for_in_u, rev_in_u = for_out_u, rev_out_u

        # v nodes processing
        for_in_v = v_seq
        rev_in_v = torch.flip(v_seq, dims=[0])
        for l in range(self.num_layers):
            for_out_v, _ = self.for_RNNs_v[l](for_in_v)
            rev_out_v, _ = self.rev_RNNs_v[l](rev_in_v)
            for_in_v, rev_in_v = for_out_v, rev_out_v

        # concatenate last snapshot embeddings
        emb_u = torch.cat([for_out_u[-1], rev_out_u[-1]], dim=-1)  # [137 x 2*d]
        emb_v = torch.cat([for_out_v[-1], rev_out_v[-1]], dim=-1)  # [1218 x 2*d]

        return emb_u, emb_v

class DDNE_Dec(nn.Module):
    '''
    BIDDNE Decoder.
    '''
    def __init__(self):
        super(DDNE_Dec, self).__init__()

    def forward(self, emb_u, emb_v):
        # emb_u: [137 x d], emb_v: [1218 x d]
        # Matrix reconstruction: [137 x 1218]
        adj_est = torch.sigmoid(torch.matmul(emb_u, emb_v.T))
        return adj_est
