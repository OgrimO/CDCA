#############################################################################
## Adapted from UGBA Unnoticeable Backdoor Attacks on Graph Neural Networks##
#############################################################################
 
from random import random
from collections import deque
import torch 
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from models.construct import model_construct


def max_norm(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def obtain_attach_nodes(args,node_idxs, size):
    size = min(len(node_idxs),size)
    rs = np.random.RandomState(args.seed)
    choice = np.arange(len(node_idxs))
    rs.shuffle(choice)
    return node_idxs[choice[:size]]

def obtain_attach_nodes_by_cluster(args,y_pred,model,node_idxs,x,labels,device,size):
    dis_weight = args.dis_weight
    cluster_centers = model.cluster_centers_
    distances = [] 
    distances_tar = []
    for id in range(x.shape[0]):
        tmp_center_label = y_pred[id]
        tmp_tar_label = args.target_class
        
        tmp_center_x = cluster_centers[tmp_center_label]
        tmp_tar_x = cluster_centers[tmp_tar_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        dis_tar = np.linalg.norm(tmp_tar_x - x[id].cpu().numpy())
        distances.append(dis)
        distances_tar.append(dis_tar)
        
    distances = np.array(distances)
    distances_tar = np.array(distances_tar)
    label_list = np.unique(y_pred)
    labels_dict = {}
    for i in label_list:
        labels_dict[i] = np.where(y_pred==i)[0]
        labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))

    each_selected_num = int(size/len(label_list)-1)
    last_seleced_num = size - each_selected_num*(len(label_list)-2)
    candidate_nodes = np.array([])
    for label in label_list:
        if(label == args.target_class):
            continue
        single_labels_nodes = labels_dict[label]
        single_labels_nodes = np.array(list(set(single_labels_nodes)))

        single_labels_nodes_dis = distances[single_labels_nodes]
        single_labels_nodes_dis = max_norm(single_labels_nodes_dis)
        single_labels_nodes_dis_tar = distances_tar[single_labels_nodes]
        single_labels_nodes_dis_tar = max_norm(single_labels_nodes_dis_tar)

        single_labels_dis_score = dis_weight * single_labels_nodes_dis + (-single_labels_nodes_dis_tar)
        single_labels_nid_index = np.argsort(single_labels_dis_score)
        sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
        if(label != label_list[-1]):
            candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:each_selected_num]])
        else:
            candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:last_seleced_num]])
    return candidate_nodes

from torch_geometric.utils import degree
def obtain_attach_nodes_by_cluster_degree_all(args,edge_index,y_pred,cluster_centers,node_idxs,x,size):
    dis_weight = args.dis_weight
    degrees = (degree(edge_index[0])  + degree(edge_index[1])).cpu().numpy()
    distances = [] 
    for id in range(x.shape[0]):
        tmp_center_label = y_pred[id]
        tmp_center_x = cluster_centers[tmp_center_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        distances.append(dis)

    distances = np.array(distances)
    print(y_pred)
    nontarget_nodes = np.where(y_pred!=args.target_class)[0]
    print("nontarget_nodes number",len(nontarget_nodes))
    print("node_idxs number",len(node_idxs))
    # print("nontarget_nodes",nontarget_nodes)
    # print("node_idxs",node_idxs)
    non_target_node_idxs = np.array(list(set(nontarget_nodes) & set(node_idxs)))
    print("non_target_node_idxs number",len(non_target_node_idxs))
    node_idxs = np.array(non_target_node_idxs)
    print("node_idxs number",len(node_idxs))
    candiadate_distances = distances[node_idxs]
    candiadate_degrees = degrees[node_idxs]
    print("mean degrees",np.mean(candiadate_degrees))
    candiadate_distances = max_norm(candiadate_distances)
    candiadate_degrees = max_norm(candiadate_degrees)

    dis_score = candiadate_distances + dis_weight * candiadate_degrees
    
    candidate_nid_index = np.argsort(dis_score)
    sorted_node_idex = np.array(node_idxs[candidate_nid_index])
    selected_nodes = sorted_node_idex
    
    return selected_nodes


import os
import time
from sklearn_extra import cluster
from sklearn.cluster import KMeans
# from kmeans_pytorch import kmeans, kmeans_predict

def cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device):
    encoder_modelpath = './modelpath/{}_{}_benign.pth'.format('GCN_Encoder', args.dataset)
    if(os.path.exists(encoder_modelpath)):
        # load existing benign model
        gcn_encoder = torch.load(encoder_modelpath)
        gcn_encoder = gcn_encoder.to(device)
        edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
        print("Loading {} encoder Finished!".format(args.model))
    else:
        gcn_encoder = model_construct(args,'GCN_Encoder',data,device, args.layer).to(device) 
        t_total = time.time()
        # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
        print("Length of training set: {}".format(len(idx_train)))
        gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True)
        print("Training encoder Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        # torch.save(gcn_encoder, encoder_modelpath)
        # print("Encoder saved at {}".format(encoder_modelpath))

    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y,idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    seen_node_idx = torch.concat([idx_train,unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x, train_edge_index,None).clone().detach()
    encoder_output = gcn_encoder(data.x,train_edge_index,None)
    y_pred = np.array(encoder_output.argmax(dim=1).cpu()).astype(int)
    gcn_encoder = gcn_encoder.cpu()
    kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
    kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    idx_attach = obtain_attach_nodes_by_cluster(args,y_pred,kmedoids,unlabeled_idx.cpu().tolist(),encoder_x,data.y,device,size).astype(int)
    return idx_attach

def cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device):
    selected_nodes_path = "./selected_nodes/{}/Overall/seed{}/nodes.txt".format(args.dataset,args.seed)
    # if(os.path.exists(selected_nodes_path)):
    #     print(selected_nodes_path)
    #     idx_attach = np.loadtxt(selected_nodes_path, delimiter=',').astype(int)
    #     idx_attach = idx_attach[:size]
    #     return idx_attach
    gcn_encoder = model_construct(args,'GCN_Encoder',data,device, args.layer).to(device) 
    t_total = time.time()
    # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
    print("Length of training set: {}".format(len(idx_train)))
    gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True)
    print("Training encoder Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y,idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    seen_node_idx = torch.concat([idx_train,unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x, train_edge_index,None).clone().detach()
    if(args.dataset == 'Cora' or args.dataset == 'Citeseer'):
        kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
        kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmedoids.cluster_centers_
        y_pred = kmedoids.predict(encoder_x.cpu().numpy())
    else:
        kmeans = KMeans(n_clusters=nclass,random_state=1)
        kmeans.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        y_pred = kmeans.predict(encoder_x.cpu().numpy())

    encoder_output = gcn_encoder(data.x,train_edge_index,None)
    idx_attach = obtain_attach_nodes_by_cluster_degree_all(args,train_edge_index,y_pred,cluster_centers,unlabeled_idx.cpu().tolist(),encoder_x,size).astype(int)
    selected_nodes_foldpath = "./selected_nodes/{}/Overall/seed{}".format(args.dataset,args.seed)
    if(not os.path.exists(selected_nodes_foldpath)):
        os.makedirs(selected_nodes_foldpath)
    selected_nodes_path = "./selected_nodes/{}/Overall/seed{}/nodes.txt".format(args.dataset,args.seed)
    if(not os.path.exists(selected_nodes_path)):
        np.savetxt(selected_nodes_path,idx_attach)
    else:
        idx_attach = np.loadtxt(selected_nodes_path, delimiter=',').astype(int)
    print("idx_attach number",len(idx_attach))
    idx_attach = idx_attach[:size]
    return idx_attach

def obtain_attach_nodes_by_cluster_degree_single(args,edge_index,y_pred,cluster_centers,node_idxs,x,size):
    dis_weight = args.dis_weight
    degrees = (degree(edge_index[0])  + degree(edge_index[1])).cpu().numpy()
    distances = [] 
  
    for i in range(node_idxs.shape[0]):
        id = node_idxs[i]
        tmp_center_label = y_pred[i]
        tmp_center_x = cluster_centers[tmp_center_label]
        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        distances.append(dis)
    distances = np.array(distances)
    print("y_pred",y_pred)
    print("node_idxs",node_idxs)

    candiadate_distances = distances
    candiadate_degrees = degrees[node_idxs]
    candiadate_distances = max_norm(candiadate_distances)
    candiadate_degrees = max_norm(candiadate_degrees)

    dis_score = candiadate_distances + dis_weight * candiadate_degrees
    candidate_nid_index = np.argsort(dis_score)
    sorted_node_idex = np.array(node_idxs[candidate_nid_index])
    selected_nodes = sorted_node_idex
    print("selected_nodes",sorted_node_idex,selected_nodes)
    return selected_nodes

def cluster_degree_selection_seperate_fixed(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device):
    gcn_encoder = model_construct(args,'GCN_Encoder',data,device, args.layer).to(device) 
    t_total = time.time()
    print("Length of training set: {}".format(len(idx_train)))
    gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True)
    print("Training encoder Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y,idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))

    seen_node_idx = torch.concat([idx_train,unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x, train_edge_index,None).clone().detach()

    encoder_output = gcn_encoder(data.x,train_edge_index,None)
    y_pred = np.array(encoder_output.argmax(dim=1).cpu()).astype(int)
    cluster_centers = []
    each_class_size = int(size/(nclass-1))
    idx_attach = np.array([])
    for label in range(nclass):
        if(label == args.target_class):
            continue

        if(label != nclass-1):
            sing_class_size = each_class_size
        else:
            last_class_size= size - len(idx_attach)
            sing_class_size = last_class_size
        idx_sing_class = (y_pred == label).nonzero()[0]
        print("idx_sing_class",idx_sing_class)
        if(len(idx_sing_class) == 0):
            continue
        print("current_class_size",sing_class_size)
        selected_nodes_path = "./selected_nodes/{}/Seperate/seed{}/class_{}.txt".format(args.dataset,args.seed,label)
        if(os.path.exists(selected_nodes_path)):
            print(selected_nodes_path)
            sing_idx_attach = np.loadtxt(selected_nodes_path, delimiter=',')
            print(sing_idx_attach)
            sing_idx_attach = sing_idx_attach[:sing_class_size]
            idx_attach = np.concatenate((idx_attach,sing_idx_attach))
        else:
            kmedoids = KMeans(n_clusters=2,random_state=1)
            kmedoids.fit(encoder_x[idx_sing_class].detach().cpu().numpy())
            sing_center = kmedoids.cluster_centers_
            cluster_ids_x = kmedoids.predict(encoder_x[idx_sing_class].cpu().numpy())
            cand_idx_sing_class = np.array(list(set(unlabeled_idx.cpu().tolist())&set(idx_sing_class)))
            if(label != nclass - 1):
                sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args,train_edge_index,cluster_ids_x,sing_center,cand_idx_sing_class,encoder_x,each_class_size).astype(int)
                # selected_nodes_foldpath = "./selected_nodes/{}/Seperate/seed{}".format(args.dataset,args.seed)
                if(not os.path.exists(selected_nodes_foldpath)):
                    os.makedirs(selected_nodes_foldpath)
                # selected_nodes_path = "./selected_nodes/{}/Seperate/seed{}/class_{}.txt".format(args.dataset,args.seed,label)
                if(not os.path.exists(selected_nodes_path)):
                    np.savetxt(selected_nodes_path,sing_idx_attach)
                else:
                    sing_idx_attach = np.loadtxt(selected_nodes_path, delimiter=',')
                sing_idx_attach = sing_idx_attach[:each_class_size]
            else:
                last_class_size= size - len(idx_attach)
                sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args,train_edge_index,cluster_ids_x,sing_center,cand_idx_sing_class,encoder_x,last_class_size).astype(int)
                # selected_nodes_path = "./selected_nodes/{}/Seperate/seed{}/class_{}.txt".format(args.dataset,args.seed,label)
                np.savetxt(selected_nodes_path,sing_idx_attach)
                if(not os.path.exists(selected_nodes_path)):
                    np.savetxt(selected_nodes_path,sing_idx_attach)
                else:
                    sing_idx_attach = np.loadtxt(selected_nodes_path, delimiter=',')
                sing_idx_attach = sing_idx_attach[:each_class_size]
            idx_attach = np.concatenate((idx_attach,sing_idx_attach))

    return idx_attach


def get_k_hop_neighbors(target_nodes, edge_index, k=2, num_nodes=None):
    if isinstance(target_nodes, torch.Tensor):
        target_nodes = target_nodes.cpu().numpy()
    target_nodes = set(target_nodes)
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    adj_list = {}
    for i in range(num_nodes):
        adj_list[i] = []
    edge_index_np = edge_index.cpu().numpy()
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        if src < num_nodes and dst < num_nodes:
            adj_list[src].append(dst)
            adj_list[dst].append(src)
    visited = set(target_nodes)
    current_level = set(target_nodes)
    for hop in range(k):
        next_level = set()
        for node in current_level:
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.add(neighbor)
        current_level = next_level
    return visited

def target_cluster_distance_selection(args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx, train_edge_index, size, device):

    print("Using target_cluster_distance selection method...")

    encoder_modelpath = './modelpath/{}_{}_benign.pth'.format('GCN_Encoder', args.dataset)
    if os.path.exists(encoder_modelpath):
        gcn_encoder = torch.load(encoder_modelpath)
        gcn_encoder = gcn_encoder.to(device)
        print("Loading {} encoder Finished!".format(args.model))
    else:
        gcn_encoder = model_construct(args, 'GCN_Encoder', data, device,layer=args.layer).to(device)
        t_total = time.time()
        print("Length of training set: {}".format(len(idx_train)))
        gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val, train_iters=args.epochs, verbose=False)
        print("Training encoder Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    encoder_x = gcn_encoder.get_h(data.x, train_edge_index, None).clone().detach()
    encoder_x = encoder_x.cpu().numpy()

    unlabeled_np = unlabeled_idx.cpu().numpy()
    candidate_nodes = np.array([nid for nid in unlabeled_np if data.y[nid].item() != args.target_class])
    print(f"Initial candidate nodes (exclude target class): {len(candidate_nodes)}")
    
    if len(candidate_nodes) == 0:
        print("Warning: No candidate nodes remain after filtering, falling back to random selection")
        return obtain_attach_nodes(args, unlabeled_idx, size)
    
    if len(candidate_nodes) < size:
        print(f"Warning: Only {len(candidate_nodes)} candidate nodes available, requested {size}")
        size = len(candidate_nodes)
    
    num_nodes = data.x.shape[0]
    edge_index_np = train_edge_index.cpu().numpy()
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        if src < num_nodes and dst < num_nodes:
            adj_list[src].append(dst)
            adj_list[dst].append(src)
    
    def get_local_cluster(node, k=2):
        visited = {node}
        current = {node}
        for _ in range(k):
            next_level = set()
            for n in current:
                for nbr in adj_list[n]:
                    if nbr not in visited:
                        visited.add(nbr)
                        next_level.add(nbr)
            current = next_level
        return visited
    
    distances = []
    for node in candidate_nodes:
        cluster_nodes = get_local_cluster(int(node), k=2)
        cluster_indices = list(cluster_nodes)
        if len(cluster_indices) <= 1:
            distances.append(0.0)
            continue
        cluster_embeddings = encoder_x[cluster_indices]
        cluster_center = cluster_embeddings.mean(axis=0)
        node_embedding = encoder_x[node]
        dist = np.linalg.norm(node_embedding - cluster_center)
        distances.append(dist)
    distances = np.array(distances)

    degrees = (degree(train_edge_index[0]) + degree(train_edge_index[1])).cpu().numpy()
    print("mean degrees",np.mean(degrees))
    candidate_degrees = degrees[candidate_nodes]

    distances_norm = max_norm(distances)
    degrees_norm = max_norm(candidate_degrees)

    dis_weight = args.dis_weight
    scores = distances_norm - dis_weight * degrees_norm
    # scores = distances_norm

    # print("scores",scores)
    print("distances_norm",distances_norm)
    print("degrees_norm",degrees_norm)

    sorted_idx = np.argsort(scores)[::-1]
    selected_idx = []
    selected_set = set()

    if len(selected_idx) < size:
        for idx in sorted_idx:
            if idx in selected_idx:
                continue
            selected_idx.append(idx)
            selected_set.add(int(candidate_nodes[idx]))
            if len(selected_idx) == size:
                break

    selected_idx = np.array(selected_idx[:size])
    selected_nodes = candidate_nodes[selected_idx]

    # print("selected_nodes", selected_nodes)
    # print("selected_nodes_scores", scores[selected_idx])
    # print("selected_nodes_distances", distances[selected_idx])
    # print("selected_nodes_degrees", candidate_degrees[selected_idx])
    
    print(f"Selected {len(selected_nodes)} attack nodes")
    print(f"Average distance to cluster: {distances[selected_idx].mean():.4f}")
    print(f"Average degree: {candidate_degrees[selected_idx].mean():.4f}")
    print(f"Score range: [{scores[selected_idx].min():.4f}, {scores[selected_idx].max():.4f}]")
    
    gcn_encoder = gcn_encoder.cpu()
    return selected_nodes.astype(int)