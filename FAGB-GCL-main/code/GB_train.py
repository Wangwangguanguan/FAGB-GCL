import argparse
import os.path as osp
import random
from time import perf_counter as t

import numpy as np
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, Actor, WebKB, Coauthor, AttributedGraphDataset, CitationFull, Amazon
from torch_geometric.utils import dropout_adj, subgraph
from torch_geometric.nn import GCNConv
from GB_model import Model, LogReg
from utils import knn_graph
from torch_geometric.utils import dropout_adj, homophily
from eval import label_classification

from sklearn.cluster import KMeans
from torch_geometric.utils import to_undirected, coalesce

import networkx as nx
from GB_graph_metis import GB_prop

def train(model: Model, x, edge_index, kg_edge_index, optimizer):
    model.train()
    optimizer.zero_grad()

    edge_index = dropout_adj(edge_index, p=args.drop_edge_rate)[0].long().contiguous()
    kg_edge_index = dropout_adj(kg_edge_index, p=args.drop_kg_edge_rate)[0].long().contiguous()

    data.gb_edge_index = data.gb_edge_index.long().contiguous()

    assert int(edge_index.min()) >= 0
    assert int(edge_index.max()) < x.size(0)

    gb_x, gb_edge_index, gb_assign = data.gb_x, data.gb_edge_index, data.gb_assign
    B = gb_x.size(0)

    assert gb_assign.dtype == torch.long
    assert (gb_assign >= 0).all(), "gb_assign has -1 or unassigned nodes!"
    assert int(gb_assign.max()) < B, "gb_assign contains out-of-range ball id!"

    if gb_edge_index.numel() > 0:
        assert gb_edge_index.dtype == torch.long
        assert int(gb_edge_index.min()) >= 0
        assert int(gb_edge_index.max()) < B, "gb_edge_index contains out-of-range ball id!"

    h0, h1, z0, z1 = model(
    x, edge_index, kg_edge_index,
    data.gb_x, data.gb_edge_index, data.gb_assign,
    data.attr_edge_index, data.attr_edge_weight
)
    loss = model.loss(h0, h1, z0, z1)

    loss.backward()
    optimizer.step()

    return loss.item()

def run(data, num_epochs, r):
    model = Model(dataset.num_features, args.num_hidden, args.tau1, args.tau2, args.l1, args.l2).to(device)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_p, weight_decay=args.wd_p, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_p, weight_decay=args.wd_p)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_p, weight_decay=args.wd_p)

    start = t()
    prev = start

    cnt_wait = 0
    best = 1e9
    best_t = 0
    patience = 20

    for epoch in range(1, num_epochs + 1):

        loss = train(model, data.x, data.edge_index, data.kg_edge_index, optimizer)
        now = t()
        if loss < best:
            best = loss
            cnt_wait = 0

            torch.save(model.state_dict(), f'model_{args.dataset}_{r}.pkl')

        else:
            cnt_wait += 1

        prev = now
           
        if cnt_wait == patience:
            print('Early stopping!')
            break

    print(f"Training finished. Total time: {t() - start:.4f}s")

    model.load_state_dict(torch.load('model_{args.dataset}_{r}.pkl'))

    embeds = model.embed(
    data.x, data.edge_index, data.kg_edge_index,
    data.gb_x, data.gb_edge_index, data.gb_assign,
    data.attr_edge_index, data.attr_edge_weight
)
    return embeds

def evaluate(model, embeds, data):
    model.eval()

    with torch.no_grad():
        logits = model(embeds)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs

def build_gb_coarsening(data):

    G = nx.Graph()
    N = data.num_nodes
    G.add_nodes_from(range(N))

    edges = data.edge_index.t().cpu().numpy()
    edges = np.unique(edges, axis=0)
    G.add_edges_from(edges)

    gb = GB_prop()
    GB_graph, clusters = gb.get_GB_graph(G)          # clusters: list[list[node_id]]
    gb_x_np = gb.transform_features(data.x.cpu().numpy(), clusters)  # [B, F]
    gb_x = torch.tensor(gb_x_np, dtype=data.x.dtype)

    gb_assign = torch.full((N,), -1, dtype=torch.long)
    for cid, nodes in enumerate(clusters):
        gb_assign[nodes] = cid

    assert (gb_assign >= 0).all(), "Some nodes were not assigned to any granular-ball!"
    assert gb_assign.max().item() < len(clusters)

    e = np.array(list(GB_graph.edges()), dtype=np.int64)
    if e.size == 0:
        gb_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        row = np.concatenate([e[:, 0], e[:, 1]])
        col = np.concatenate([e[:, 1], e[:, 0]])
        gb_edge_index = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long)

    return gb_x, gb_edge_index, gb_assign

def build_kmeans_attr_graph(x, edge_index, n_clusters=50, intra_k=10, lambda_attr=0.1):

    x_cpu = x.detach().cpu()
    N = x_cpu.size(0)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(x_cpu.numpy())
    labels = torch.from_numpy(labels).long()  # [N]

    x_norm = F.normalize(x_cpu, p=2, dim=1)

    src_list, dst_list, w_list = [], [], []
    for c in range(n_clusters):
        idx = (labels == c).nonzero(as_tuple=False).view(-1)
        m = idx.numel()
        if m <= 1:
            continue

        feats = x_norm[idx]         # [m, F]
        sim = feats @ feats.t()     # [m, m]
        sim.fill_diagonal_(-1.0)

        kk = min(intra_k, m - 1)
        vals, nn_local = torch.topk(sim, k=kk, dim=1)

        src = idx.repeat_interleave(kk)
        dst = idx[nn_local.reshape(-1)]

        w = (vals.reshape(-1).clamp(min=0.0)) * lambda_attr

        src_list.append(src)
        dst_list.append(dst)
        w_list.append(w)

    if len(src_list) == 0:
        edge_sim = torch.empty((2, 0), dtype=torch.long)
        w_sim = torch.empty((0,), dtype=x_cpu.dtype)
    else:
        src = torch.cat(src_list)
        dst = torch.cat(dst_list)
        w_sim = torch.cat(w_list).to(dtype=x_cpu.dtype)
        edge_sim = torch.stack([src, dst], dim=0).long()

    edge_orig = edge_index.detach().cpu().long()
    w_orig = torch.ones(edge_orig.size(1), dtype=x_cpu.dtype)

    edge_all = torch.cat([edge_orig, edge_sim], dim=1)
    w_all = torch.cat([w_orig, w_sim], dim=0)

    edge_all, w_all = to_undirected(edge_all, w_all, num_nodes=N)
    edge_all, w_all = coalesce(edge_all, w_all, N, N)

    return edge_all.long(), w_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--SEED', type=int, default=0)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--lr_p', type=float, default=0.001)
    parser.add_argument('--lr_m', type=float, default=0.01)
    parser.add_argument('--wd_p', type=float, default=0.0)
    parser.add_argument('--wd_m', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--tau1', type=float, default=1.1)
    parser.add_argument('--tau2', type=float, default=1.1)
    parser.add_argument('--l1', type=float, default=1.0)
    parser.add_argument('--l2', type=float, default=1.0)
    parser.add_argument('--metric', type=str, default='cosine')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--drop_edge_rate', type=float, default=0.0)
    parser.add_argument('--drop_kg_edge_rate', type=float, default=0.0)
    args = parser.parse_args()
    print("=====", args.dataset,"=====")
    def get_dataset(path, name):
        if name in ['cora', 'citeseer', 'pubmed']:
            return Planetoid(path, name, transform=T.NormalizeFeatures())
        elif name in ['Cornell', 'Texas', 'Wisconsin']:
            return WebKB(path, name)
        elif name in ['Actor']:
            return Actor(path, transform=T.NormalizeFeatures())
        elif name in ['DBLP']:
            return CitationFull(path, name, transform=T.NormalizeFeatures())
        elif name in ['CS']:
            return Coauthor(path, name, transform=T.NormalizeFeatures())
        elif name in ['Photo']:
            return Amazon(path, name, transform=T.NormalizeFeatures())
        
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    np.random.seed(args.SEED)
    torch.backends.cudnn.deterministic = True

    path = osp.join(osp.expanduser('..'), 'data', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    if not hasattr(data, 'train_mask'):
        x = data.x
        k = int(x.size(0) * 0.1)
        idx = torch.arange(x.size(0))
        idx = idx[torch.randperm(idx.size(0))[:k]]
        data.train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        data.train_mask[idx] = True
        
        remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]
        data.val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        data.val_mask[remaining[:k]] = True
        
        data.test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        data.test_mask[remaining[k:]] = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if osp.exists(f'../knn/{args.dataset}_{args.K}_knn_graph.pt'):
        print("knn graph exists")
    else:
        torch.save(knn_graph(data.x, k=args.K, metric=args.metric), f'../knn/{args.dataset}_{args.K}_knn_graph.pt')
    
    data.kg_edge_index = torch.load(f'../knn/{args.dataset}_{args.K}_knn_graph.pt')
    print("knn graph built")

    data.gb_x, data.gb_edge_index, data.gb_assign = build_gb_coarsening(data)

    data.attr_edge_index, data.attr_edge_weight = build_kmeans_attr_graph(
        data.x, data.edge_index,
        n_clusters=50,
        intra_k=10,
        lambda_attr=0.1
    )

    data = data.to(device)

    data.attr_edge_index = data.attr_edge_index.to(device)
    data.attr_edge_weight = data.attr_edge_weight.to(device)

    f_accs = []
    for r in range(10):
        
        if len(data.train_mask.size())>1:
            data.train_mask = data.train_mask[:, r]
            data.val_mask = data.val_mask[:, r]
            data.test_mask = data.test_mask[:, r]
        embeds = run(data, args.num_epochs, r)
        train_lbls = data.y[data.train_mask]
        test_lbls = data.y[data.test_mask]
        test_acc = label_classification(embeds, data) *100
        f_accs.append(test_acc)

    print(np.mean(f_accs), np.std(f_accs))
       
  