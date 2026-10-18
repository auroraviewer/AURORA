import torch
import torch.nn as nn
import torch.nn.functional as F
class AURORA(nn.Module):
    def __init__(self, input_size, output_size, device, 
                 celltype_num, bulk_size = 1212, latent_size=256,
                 bidirectional=False,
                 use_bulk=True,
                 pred_celltype=True,
                 finetune=False):
        super(AURORA, self).__init__()
        self.input_size = input_size
        self.bulk_size = bulk_size
        self.device = device
        self.bidirectional = bidirectional
        self.use_bulk = use_bulk
        self.pred_celltype = pred_celltype
        major_num = celltype_num
        self.finetune = finetune
        self.latent_size = latent_size
        #TODO: Try using self-attention for bulk_encoder as in Geneformer (or simply use their model then do another Feedforward)
        self.bulk_encoder = nn.Sequential(
            nn.Linear(bulk_size, latent_size),
            nn.LeakyReLU()
        )
        self.bulk_encoder.to(device)
        self.lstm = nn.LSTM(input_size, latent_size, batch_first=True, device=device, bidirectional=self.bidirectional)
        if not finetune:
            if self.bidirectional:
                self.decoder = nn.Sequential(
                    nn.Linear(latent_size*2, latent_size),
                    nn.LeakyReLU(),
                    nn.Linear(latent_size, latent_size//2),
                    nn.LeakyReLU(),
                    nn.Linear(latent_size//2, latent_size),
                    nn.LeakyReLU(),
                    nn.Linear(latent_size, output_size),
                    #nn.ELU()
                    nn.Sigmoid()
                )
            else:
                self.decoder = nn.Sequential(
                    nn.Linear(latent_size, latent_size//2),
                    nn.LeakyReLU(),
                    nn.Linear(latent_size//2, latent_size),
                    nn.LeakyReLU(),
                    nn.Linear(latent_size, output_size),
                    #nn.ELU()
                    nn.Sigmoid()
                )
            self.decoder.to(device)
        else:
            self.new_decoder = nn.Sequential(
            nn.Linear(latent_size, latent_size//2),
            nn.LeakyReLU(),
            nn.Linear(latent_size//2, latent_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size, output_size),
            #nn.ELU()
            nn.Sigmoid()
        )
            self.new_decoder.to(device)
        self.celltype_decoder = nn.Sequential(
            nn.Linear(latent_size, max(latent_size//4, 32)),
            nn.LeakyReLU(),
            nn.Linear(max(latent_size//4, 32), 32),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(32, major_num),
            #nn.ELU()
            # nn Clamp(min=-5, max=0),
            nn.LogSoftmax(dim=-1)
        )
        self.celltype_decoder.to(device)
    #
    def forward(self, emb_hierrachy, bulk, return_latent_emb=False, highres=False, celltype_only=False):
        if self.bidirectional:
            d = 2
        else:
            d = 1
        if not highres:
            if self.use_bulk:
                bulk_emb = self.bulk_encoder(bulk)
                hierrachy_emb, _ = self.lstm(emb_hierrachy,
                                            (bulk_emb.repeat(d,emb_hierrachy.shape[0],1),
                                            torch.zeros(d,emb_hierrachy.shape[0],self.latent_size,device=self.device)))
            else:
                hierrachy_emb, _ = self.lstm(emb_hierrachy)
        else:
            hierrachy_emb, _ = self.lstm(emb_hierrachy) #,
                                            # (self.tmp_encoder(bulk), #bulk_emb.repeat(d,emb_hierrachy.shape[0],1),
                                            # torch.zeros(d,emb_hierrachy.shape[0],self.latent_size,device=self.device)))
        #
        if hierrachy_emb.ndim == 3:
            hierrachy_emb = torch.unbind(hierrachy_emb, dim=1)
        else:
            hierrachy_emb = [hierrachy_emb]
        # ``celltype_only`` skips the (frozen) gene-expression decoder so the
        # cell-type training phase only pays for the proportion head.
        if not celltype_only:
            if not self.finetune:
                results = [self.decoder(s) for s in hierrachy_emb]
            else:
                results = [self.new_decoder(s) for s in hierrachy_emb]
            output = torch.stack(results, dim=1)
        else:
            output = None
        if self.pred_celltype:
            major_prop = [self.celltype_decoder(hierrachy_emb[s]) for s in range(len(hierrachy_emb))]
            major_prop = torch.stack(major_prop, dim=1)
        else:
            major_prop = None
        #
        if return_latent_emb:
            return output, major_prop, hierrachy_emb
        else:
            return output, major_prop

        
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True,
                 cond_dim=None, bulk_size=None):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.cond_dim = cond_dim
        self.cond_hidden = bulk_size
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        if self.cond_dim is not None:
            self.condition_encoder = nn.Sequential(
                nn.Linear(bulk_size, cond_dim),
                nn.LeakyReLU()
            )
            attn_input_dim = 2 * out_features + cond_dim
        else:
            self.condition_encoder = None
            attn_input_dim = 2 * out_features
        self.a = nn.Parameter(torch.zeros(size=(attn_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, h, edge_index, cond=None):
        """
        h:          (N, in_features)
        edge_index: (2, E) where edge_index[0]=target_i, edge_index[1]=source_j
                    (matches old adj[i, j] semantics)
        cond:       optional condition tensor
        """
        Wh = torch.mm(h, self.W)  # (N, out_features)
        N = Wh.size(0)

        row, col = edge_index[0].long(), edge_index[1].long()  # i, j
        e = self._prepare_edge_attention_scores(Wh, row, col, cond)  # (E,)

        # sparse softmax over neighbors j for each i (row-wise)
        max_per_row = torch.full((N,), float("-inf"), device=Wh.device, dtype=Wh.dtype)
        max_per_row.scatter_reduce_(0, row, e, reduce="amax", include_self=True)
        exp_e = torch.exp(e - max_per_row[row])

        denom = torch.zeros(N, device=Wh.device, dtype=Wh.dtype)
        denom.index_add_(0, row, exp_e)
        attention = exp_e / (denom[row] + 1e-16)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # h_prime[i] = sum_j attention(i,j) * Wh[j]
        h_prime = torch.zeros_like(Wh)
        h_prime.index_add_(0, row, attention.unsqueeze(1) * Wh[col])

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_edge_attention_scores(self, Wh, row, col, cond=None):
        """
        Compute e_ij only on edges in edge_index.
        """
        edge_parts = [Wh[row], Wh[col]]  # (E, out_features), (E, out_features)

        if self.condition_encoder is not None:
            if cond is None:
                raise ValueError("cond must be provided when cond_dim is set.")
            cond = cond.to(Wh.device)
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)
            if cond.dim() != 2:
                raise ValueError("cond must be a 1D or 2D tensor.")

            cond_emb = self.condition_encoder(cond)  # (1, cond_dim) or (N, cond_dim)
            N = Wh.size(0)
            if cond_emb.shape[0] == 1:
                cond_emb = cond_emb.expand(N, -1)
            elif cond_emb.shape[0] != N:
                raise ValueError(
                    f"Condition batch size {cond_emb.shape[0]} must be 1 or match node count {N}."
                )
            edge_parts.append(cond_emb[row])  # align with target node i

        edge_feat = torch.cat(edge_parts, dim=1)     # (E, attn_input_dim)
        e = torch.matmul(edge_feat, self.a).squeeze(1)  # (E,)
        return self.leakyrelu(e)
    
class GAT(nn.Module):
    def __init__(self, ninput, noutput, celltype_num, nhid=256,  dropout=0.2, alpha=0.2,
                 cond_dim=None, bulk_size=None):
        super(GAT, self).__init__()
        self.cond_dim = cond_dim
        self.cond_hidden = bulk_size
        self.gat_layer1 = GraphAttentionLayer(ninput, nhid, dropout, alpha,
                                              cond_dim=cond_dim, bulk_size=bulk_size)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.gat_layer2 = GraphAttentionLayer(nhid, nhid, dropout, alpha,
                                              cond_dim=cond_dim, bulk_size=bulk_size)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.gat_layer3 = GraphAttentionLayer(nhid, nhid, dropout, alpha, concat=False,
                                              cond_dim=cond_dim, bulk_size=bulk_size)
        self.decoder = nn.Sequential(
                    nn.Linear(nhid, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, noutput),
                    #nn.ELU()
                    nn.Sigmoid()
                )
        self.celltype_decoder = nn.Sequential(
            nn.Linear(nhid, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(32, celltype_num),
            #nn.ELU()
            # nn Clamp(min=-5, max=0),
            nn.LogSoftmax(dim=-1)
        )
    def forward(self, x, edge_index, bulk=None):
        x = F.elu(self.gat_layer1(x, edge_index, cond=bulk))
        x = self.bn1(x)
        x = F.elu(self.gat_layer2(x, edge_index, cond=bulk))
        x = self.bn2(x)
        x = self.gat_layer3(x, edge_index, cond=bulk)
        output = self.decoder(x)
        major_prop = self.celltype_decoder(x)
        return output, major_prop

# NOTE (aurora 2.0 reorganisation): torch_geometric is only needed by the GCN
# and GraphLSTM variants, never by the AURORA model used for inference.  It used
# to be imported at module import time, which made it a hard dependency of every
# inference run.  It is now resolved lazily, and `model.GCNConv` /
# `model.GraphConv` still work through the module __getattr__ below.
# Original statement preserved for reference:
#   from torch_geometric.nn import GraphConv, GCNConv


def _torch_geometric_nn():
    """Import torch_geometric on demand and return ``(GraphConv, GCNConv)``."""
    from torch_geometric.nn import GraphConv, GCNConv

    return GraphConv, GCNConv


def __getattr__(name):
    if name in ("GraphConv", "GCNConv"):
        GraphConv, GCNConv = _torch_geometric_nn()
        value = GraphConv if name == "GraphConv" else GCNConv
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class GCN(nn.Module):
    """
    3-layer GCN for gene expression and cell-type prediction.
    Mirrors the GAT architecture but uses GCNConv (Kipf & Welling).
    """
    def __init__(self, ninput, noutput, celltype_num, nhid=256, dropout=0.2,
                 cond_dim=None, bulk_size=None):
        super(GCN, self).__init__()
        GCNConv = _torch_geometric_nn()[1]
        self.cond_dim = cond_dim
        self.cond_hidden = bulk_size
        self.dropout = dropout

        if cond_dim is not None and bulk_size is not None:
            self.condition_encoder = nn.Sequential(
                nn.Linear(bulk_size, cond_dim),
                nn.LeakyReLU()
            )
            self.conv1 = GCNConv(ninput + cond_dim, nhid)
            self.conv2 = GCNConv(nhid + cond_dim, nhid)
            self.conv3 = GCNConv(nhid + cond_dim, nhid)
        else:
            self.condition_encoder = None
            self.conv1 = GCNConv(ninput, nhid)
            self.conv2 = GCNConv(nhid, nhid)
            self.conv3 = GCNConv(nhid, nhid)

        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        self.decoder = nn.Sequential(
            nn.Linear(nhid, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, noutput),
            nn.Sigmoid()
        )
        self.celltype_decoder = nn.Sequential(
            nn.Linear(nhid, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, celltype_num),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, edge_index, bulk=None):
        """
        x:          (N, ninput)  node features
        edge_index: (2, E)       edge list
        bulk:       (1, bulk_size) or (N, bulk_size) optional condition
        Returns:
            output:     (N, noutput)   gene expression predictions
            major_prop: (N, celltype_num) cell-type proportion log-probs
        """
        if self.condition_encoder is not None:
            if bulk is None:
                raise ValueError("bulk must be provided when cond_dim is set.")
            cond = bulk.to(x.device)
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)
            cond_emb = self.condition_encoder(cond)  # (1, cond_dim) or (N, cond_dim)
            N = x.shape[0]
            if cond_emb.shape[0] == 1:
                cond_emb = cond_emb.expand(N, -1)

            # Layer 1
            x = torch.cat([x, cond_emb], dim=1)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.bn1(x)
            x = F.dropout(x, self.dropout, training=self.training)

            # Layer 2
            x = torch.cat([x, cond_emb], dim=1)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.bn2(x)
            x = F.dropout(x, self.dropout, training=self.training)

            # Layer 3 (no concat=False equivalent, just no BN/dropout)
            x = torch.cat([x, cond_emb], dim=1)
            x = self.conv3(x, edge_index)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.bn1(x)
            x = F.dropout(x, self.dropout, training=self.training)

            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.bn2(x)
            x = F.dropout(x, self.dropout, training=self.training)

            x = self.conv3(x, edge_index)

        output = self.decoder(x)
        major_prop = self.celltype_decoder(x)
        return output, major_prop


class GraphLSTM(nn.Module):
    def __init__(self, input_size, output_size, device, 
                 celltype_num, bulk_size = 1212, 
                 pred_celltype=True):
        super(GraphLSTM, self).__init__()
        self.input_size = input_size
        self.bulk_size = bulk_size
        self.device = device
        self.pred_celltype = pred_celltype
        major_num = celltype_num
        #TODO: Try using self-attention for bulk_encoder as in Geneformer (or simply use their model then do another Feedforward)
        self.bulk_encoder = nn.Sequential(
            nn.Linear(bulk_size, 256),
            nn.LeakyReLU()
        )
        self.bulk_encoder.to(device)
        GraphConv = _torch_geometric_nn()[0]
        self.conv = GraphConv(input_size, 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True, device=device)
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size),
            #nn.ELU()
            nn.Sigmoid()
        )
        self.decoder.to(device)
        self.celltype_decoder = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(32, major_num),
            #nn.ELU()
            # nn Clamp(min=-5, max=0),
            nn.LogSoftmax(dim=-1)
        )
        self.celltype_decoder.to(device)
    #
    def forward(self, emb_hierrachy, bulk, edge_index_list):
        d = 1
        bulk_emb = self.bulk_encoder(bulk)
        emb_after_gcn = [self.conv(emb_hierrachy[:, i, :], edge_index_list[i]) for i in range(emb_hierrachy.shape[1])]
        emb_after_gcn = torch.stack(emb_after_gcn, dim=1)
        hierrachy_emb, _ = self.lstm(emb_after_gcn,
                                    (bulk_emb.repeat(d,emb_after_gcn.shape[0],1),
                                    torch.zeros(d,emb_after_gcn.shape[0],256,device=self.device)))
        #
        hierrachy_emb = torch.unbind(hierrachy_emb, dim=1)
        results = [self.decoder(s) for s in hierrachy_emb]
        output = torch.stack(results, dim=1)
        if self.pred_celltype:
            major_prop = [self.celltype_decoder(hierrachy_emb[s]) for s in range(len(hierrachy_emb))]
            major_prop = torch.stack(major_prop, dim=1)
        else:
            major_prop = None
        #
        return output, major_prop

