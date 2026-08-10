import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj,dense_to_sparse
import torch
import scipy.sparse as sp
from models.reconstruct import MLPAE


def edge_sim_analysis(edge_index, features):
    sims = []
    for (u,v) in edge_index:
        sims.append(float(F.cosine_similarity(features[u].unsqueeze(0),features[v].unsqueeze(0))))
    sims = np.array(sims)
    # print(f"mean: {sims.mean()}, <0.1: {sum(sims<0.1)}/{sims.shape[0]}")
    return sims

def prune_unrelated_edge(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            # print(edge_sim1)
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
        # edge_sims = edge_sims.to(device)
    else:
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    updated_edge_index = edge_index[:,edge_sims>args.prune_thr]
    updated_edge_weights = edge_weights[edge_sims>args.prune_thr]
    return updated_edge_index,updated_edge_weights

def prune_unrelated_edge_isolated(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)

    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            # print(edge_sim1)
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
        # edge_sims = edge_sims.to(device)
    else:
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])

    dissim_edges_index = np.where(edge_sims.cpu()<=args.prune_thr)[0]
    edge_weights[dissim_edges_index] = 0

    dissim_edges = edge_index[:,dissim_edges_index]
    dissim_nodes = torch.cat([dissim_edges[0],dissim_edges[1]]).tolist()
    dissim_nodes = list(set(dissim_nodes))

    updated_edge_index = edge_index[:,edge_weights>0.0]
    updated_edge_weights = edge_weights[edge_weights>0.0]
    return updated_edge_index,updated_edge_weights,dissim_nodes 

def select_target_nodes(args,seed,model,features,edge_index,edge_weights,labels,idx_val,idx_test):
    test_ca,test_correct_index = model.test_with_correct_nodes(features,edge_index,edge_weights,labels,idx_test)
    test_correct_index = test_correct_index.tolist()

    test_correct_nodes = idx_test[test_correct_index].tolist()

    target_class_nodes_test = [int(nid) for nid in idx_test
            if labels[nid]==args.target_class] 

    idx_val,idx_test = idx_val.tolist(),idx_test.tolist()
    rs = np.random.RandomState(seed)
    cand_atk_test_nodes = list(set(test_correct_nodes) - set(target_class_nodes_test))
    atk_test_nodes = rs.choice(cand_atk_test_nodes, args.target_test_nodes_num)

    cand_clean_test_nodes = list(set(idx_test) - set(atk_test_nodes))
    clean_test_nodes = rs.choice(cand_clean_test_nodes, args.clean_test_nodes_num)

    N = features.shape[0]
    cand_poi_train_nodes = list(set(idx_val)-set(atk_test_nodes)-set(clean_test_nodes))
    poison_nodes_num = int(N * args.vs_ratio)
    poi_train_nodes = rs.choice(cand_poi_train_nodes, poison_nodes_num)
    
    return atk_test_nodes, clean_test_nodes,poi_train_nodes

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()


def clu_prune_unrelated_edge(args,edge_index,edge_weights,x,device,large_graph=True):
    # edge_index = edge_index[:,edge_weights>0.0].to(device)
    # edge_weights = edge_weights[edge_weights>0.0].to(device)
    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)
    x = x.to(device)

    kmeans = faiss.Kmeans(x.shape[1], 2, niter=20) 
    kmeans.train(x.cpu().detach().numpy())
    centroids = torch.FloatTensor(kmeans.centroids).to(x.device)
    D, I = kmeans.index.search(x.cpu().detach().numpy(), 1)
    cluster_counts = np.bincount(I.squeeze())

    dominant_cluster = np.argmax(cluster_counts)

    last = len(edge_weights)-1

    indomain = []
    for idx, label in enumerate(I):
        if label[0] == dominant_cluster:
            indomain.append(idx)
    # print(indomain)

    indomain_tensor = torch.tensor(indomain, device=edge_index.device)
    mask = torch.isin(edge_index, indomain_tensor).all(dim=0)
    # print("Mask:", mask)

    updated_edge_index = edge_index[:, mask]
    updated_edge_weights = edge_weights[mask]
    # updated_edge_index = edge_index[:, :-2]
    # updated_edge_weights = edge_weights[:-2]

    print("Updated edge_index:", updated_edge_index)
    print("Updated edge_weights:", updated_edge_weights)
    # updated_edge_index = edge_index[:,:-1]
    # updated_edge_weights = edge_weights[:-1]
    return updated_edge_index,updated_edge_weights
    
def reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,ori_x,ori_edge_index,device, idx, large_graph=True):
    poison_x = poison_x.to(device)
    AE = MLPAE(poison_x, poison_x[len(ori_x):], device, args.rec_epochs)
    AE.fit()
    rec_score_ori = AE.inference(poison_x)
    threshold = np.percentile(rec_score_ori.detach().cpu().numpy(), 97)
    mask = rec_score_ori>threshold
    keep_edges_mask = ~(mask[poison_edge_index[0]] | mask[poison_edge_index[1]])
    filtered_poison_edge_index = poison_edge_index[:, keep_edges_mask]
    filtered_poison_edge_weights = poison_edge_weights[keep_edges_mask]
    return filtered_poison_edge_index,filtered_poison_edge_weights
