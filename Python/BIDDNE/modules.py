import torch
import torch.nn as nn

class DDNE(nn.Module):
    '''
    Clase DDNE adaptada para grafos bipartitos

    Separamos los nodos U de los V, generamos embeddings distintos segun el tipo de nodo;
    Tenemos una bidireccionalidad temporal que en teoria deberia capturar mejor los patrones temporales
    Reconstruimos explicitamente la matriz bipartita
    '''
    def __init__(self, enc_dims, dec_dims, dropout_rate, num_U, num_V):
        super(DDNE, self).__init__()
        self.enc_dims = enc_dims
        self.dec_dims = dec_dims
        self.dropout_rate = dropout_rate
        
        #Encoder bipartito requiere info de número de nodos en cada conjunto
        self.enc = DDNE_Enc(self.enc_dims, self.dropout_rate, num_U, num_V)
        self.dec = DDNE_Dec(self.dec_dims, self.dropout_rate)

    def forward(self, adj_list):
        '''
        :param adj_list: lista de matrices de adyacencia bipartita históricas [num_U, num_V]
        :return: matriz estimada y embeddings dinámicos separados (U, V)
        '''
        dyn_emb_U, dyn_emb_V = self.enc(adj_list)
        adj_est = self.dec(dyn_emb_U, dyn_emb_V)
        return adj_est, (dyn_emb_U, dyn_emb_V)

class DDNE_Enc(nn.Module):
    '''
    Encoder de DDNE adaptado para grafos bipartitos
    '''
    def __init__(self, enc_dims, dropout_rate, num_U, num_V):
        super(DDNE_Enc, self).__init__()
        self.enc_dims = enc_dims
        self.dropout_rate = dropout_rate
        self.num_enc_layers = len(self.enc_dims) - 1
        
        #Listas de GRUs para nodos U (forward y reverse)
        #forward lee de pasado a presente, reverse lee de presente a pasado
        self.for_RNN_layer_list_U = nn.ModuleList()
        self.rev_RNN_layer_list_U = nn.ModuleList()
        #Listas de GRUs para nodos V (forward y reverse)
        self.for_RNN_layer_list_V = nn.ModuleList()
        self.rev_RNN_layer_list_V = nn.ModuleList()
        
        #En la primera capa, input_size es el tamaño del conjunto contrario
        input_size_U = num_V  
        input_size_V = num_U
        
        for l in range(self.num_enc_layers):
            #Para nodos U
            self.for_RNN_layer_list_U.append(
                nn.GRU(input_size=input_size_U, hidden_size=self.enc_dims[l+1]))
            self.rev_RNN_layer_list_U.append(
                nn.GRU(input_size=input_size_U, hidden_size=self.enc_dims[l+1]))
            #Para nodos V
            self.for_RNN_layer_list_V.append(
                nn.GRU(input_size=input_size_V, hidden_size=self.enc_dims[l+1]))
            self.rev_RNN_layer_list_V.append(
                nn.GRU(input_size=input_size_V, hidden_size=self.enc_dims[l+1]))
            
            #Para la siguiente capa, el input_size es el hidden_size anterior
            input_size_U = self.enc_dims[l+1]
            input_size_V = self.enc_dims[l+1]

    def forward(self, adj_list):
        '''
        :param adj_list: list of historical bipartite adjacency matrices (shape: [num_U, num_V])
        :return: dynamic embeddings for nodes U and V

        agarra la historia de las conexiones entre paises y productos. Luego los procesa mediante GRU's mirando hacia atras y hacia adelante
        para generar un embedding que sintetiza el comportamiento en esos snapshots
        '''
        win_size = len(adj_list)  # Número de snapshots
        num_U, num_V = adj_list[0].shape

        #Preparamos entradas separadas para nodos U y nodos V
        #Cada fila (adj[i]) representa las conexiones de un nodo de U hacia todos los V
        for_RNN_input_U = []
        for_RNN_input_V = []
        rev_RNN_input_U = []
        rev_RNN_input_V = []

        for i in range(win_size):
            #fwd mira la información de forma normal, de pasado a presente
            #rev mira de presente a pasado
            A_fwd = adj_list[i]                 # [num_U, num_V]
            A_rev = adj_list[win_size - 1 - i]  # [num_U, num_V]

            #Entrada para nodos U: cada nodo U es una fila con conexiones a V
            for_RNN_input_U.append(A_fwd)
            rev_RNN_input_U.append(A_rev)

            #Entrada para nodos V: cada nodo V es una columna con conexiones desde U
            for_RNN_input_V.append(A_fwd.T)
            rev_RNN_input_V.append(A_rev.T)

        #Convertimos a tensores 3D [T, N, D]
        for_RNN_input_U = torch.stack(for_RNN_input_U)  # [T, num_U, num_V]
        rev_RNN_input_U = torch.stack(rev_RNN_input_U)  # [T, num_U, num_V]
        for_RNN_input_V = torch.stack(for_RNN_input_V)  # [T, num_V, num_U]
        rev_RNN_input_V = torch.stack(rev_RNN_input_V)  # [T, num_V, num_U]

        #Aplicamos GRUs independientes para U y V
        for_RNN_output_U = for_RNN_input_U
        rev_RNN_output_U = rev_RNN_input_U
        for_RNN_output_V = for_RNN_input_V
        rev_RNN_output_V = rev_RNN_input_V

        for l in range(self.num_enc_layers):
            for_RNN_output_U, _ = self.for_RNN_layer_list_U[l](for_RNN_output_U)
            rev_RNN_output_U, _ = self.rev_RNN_layer_list_U[l](rev_RNN_output_U)

            for_RNN_output_V, _ = self.for_RNN_layer_list_V[l](for_RNN_output_V)
            rev_RNN_output_V, _ = self.rev_RNN_layer_list_V[l](rev_RNN_output_V)

        #Combinar output forward + reverse a lo largo del tiempo
        dyn_emb_U_list = []
        dyn_emb_V_list = []
        for i in range(win_size):
            emb_U_t = torch.cat((for_RNN_output_U[i], rev_RNN_output_U[i]), dim=1)  # [num_U, 2H]
            emb_V_t = torch.cat((for_RNN_output_V[i], rev_RNN_output_V[i]), dim=1)  # [num_V, 2H]
            dyn_emb_U_list.append(emb_U_t)
            dyn_emb_V_list.append(emb_V_t)

        #Concatenamos embeddings a través del tiempo: [num_nodes, 2H*T]
        dyn_emb_U = torch.cat(dyn_emb_U_list, dim=1)
        dyn_emb_V = torch.cat(dyn_emb_V_list, dim=1)

        return dyn_emb_U, dyn_emb_V
    
class DDNE_Dec(nn.Module):
    '''
    Decoder adaptado para grafos bipartitos

    Toma los embedings dinamicos de u y v, los pasa por redes neuronales (mlp), calcula el dot product entre cada par, aplica sigmoid y devuelve la matriz de predicciones
    '''
    def __init__(self, dec_dims, dropout_rate):
        super(DDNE_Dec, self).__init__()
        self.dec_dims = dec_dims
        self.dropout_rate = dropout_rate
        self.num_dec_layers = len(self.dec_dims) - 1
        
        #MLP para nodos U
        self.dec_U = nn.ModuleList()
        #MLP para nodos V
        self.dec_V = nn.ModuleList()
        for l in range(self.num_dec_layers):
            self.dec_U.append(
                nn.Linear(in_features=self.dec_dims[l], out_features=self.dec_dims[l+1]))
            self.dec_V.append(
                nn.Linear(in_features=self.dec_dims[l], out_features=self.dec_dims[l+1]))
        
        self.dec_drop = nn.ModuleList()
        for l in range(self.num_dec_layers - 1):
            self.dec_drop.append(nn.Dropout(p=self.dropout_rate))
        
    def forward(self, dyn_emb_U, dyn_emb_V):
        '''
        :param dyn_emb_U: embedding dinámico para nodos U [num_U, emb_dim]
        :param dyn_emb_V: embedding dinámico para nodos V [num_V, emb_dim]
        :return: matriz bipartita estimada [num_U, num_V]
        '''
        dec_input_U = dyn_emb_U
        dec_input_V = dyn_emb_V
        
        #Pasar por el MLP
        for l in range(self.num_dec_layers - 1):
            dec_input_U = torch.relu(self.dec_U[l](dec_input_U))
            dec_input_U = self.dec_drop[l](dec_input_U)
            dec_input_V = torch.relu(self.dec_V[l](dec_input_V))
            dec_input_V = self.dec_drop[l](dec_input_V)
        
        dec_input_U = self.dec_U[-1](dec_input_U)
        dec_input_V = self.dec_V[-1](dec_input_V)
        
        #Producto punto para reconstruir matriz bipartita
        adj_est = torch.sigmoid(torch.matmul(dec_input_U, dec_input_V.T))
        
        return adj_est
