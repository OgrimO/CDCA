#%%
import torch
import numpy as np
import warnings
from torch_geometric.utils import k_hop_subgraph

def tensor2onehot(labels):
    labels = labels.long()
    eye = torch.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx.to(labels.device)

def accuracy(output, labels):
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def idx_to_mask(indices, n):
    mask = torch.zeros(n, dtype=torch.bool)
    mask[indices] = True
    return mask
import scipy.sparse as sp
def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
# %%
def subgraph(subset,edge_index, edge_attr = None, relabel_nodes: bool = False):

    device = edge_index.device

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    # if relabel_nodes:
    #     node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
    #                            device=device)
    #     node_idx[subset] = torch.arange(subset.sum().item(), device=device)
    #     edge_index = node_idx[edge_index]


    return edge_index, edge_attr, edge_mask
# %%

def calculate_asr_by_distance(attack_nodes, edge_index, output, target_class, original_labels, device=None, num_nodes=None):
    
    if device is None:
        device = edge_index.device
    
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    valid_attack_mask = (attack_nodes >= 0) & (attack_nodes < num_nodes)
    valid_attack_nodes = attack_nodes[valid_attack_mask]
    
    if len(valid_attack_nodes) == 0:
        return {
            'attack_nodes': {'success': 0, 'total': 0, 'asr': 0.0},
            '1_hop': {'success': 0, 'total': 0, 'asr': 0.0},
            '2_hop': {'success': 0, 'total': 0, 'asr': 0.0},
            '3_hop': {'success': 0, 'total': 0, 'asr': 0.0}
        }
    
    results = {}
    original_target_mask = original_labels[valid_attack_nodes] != target_class
    filtered_attack_nodes = valid_attack_nodes[original_target_mask]
    
    if len(filtered_attack_nodes) > 0:
        attack_predictions = output.argmax(dim=1)[filtered_attack_nodes]
        attack_success = (attack_predictions == target_class).sum().item()
        results['attack_nodes'] = {
            'success': attack_success,
            'total': len(filtered_attack_nodes),
            'asr': attack_success / len(filtered_attack_nodes) if len(filtered_attack_nodes) > 0 else 0.0
        }
    else:
        results['attack_nodes'] = {
            'success': 0,
            'total': 0,
            'asr': 0.0
        }

    try:
        adj_list = {}
        for i in range(num_nodes):
            adj_list[i] = []
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src < num_nodes and dst < num_nodes:
                adj_list[src].append(dst)
                adj_list[dst].append(src)

        distances = [-1] * num_nodes
        queue = []

        for node in valid_attack_nodes:
            node_idx = node.item()
            if node_idx < num_nodes:
                distances[node_idx] = 0
                queue.append(node_idx)

        while queue:
            current = queue.pop(0)
            current_dist = distances[current]
            
            if current_dist < 3:
                for neighbor in adj_list[current]:
                    if distances[neighbor] == -1:
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)

        original_graph_size = len(original_labels)
        for k in [1, 2, 3]:
            k_hop_nodes = [i for i in range(num_nodes) if distances[i] == k]
            k_hop_nodes = [i for i in k_hop_nodes if original_labels[i] != target_class]
            k_hop_nodes = [i for i in k_hop_nodes if i < original_graph_size]
            
            if len(k_hop_nodes) > 0:
                k_hop_tensor = torch.tensor(k_hop_nodes, device=device)
                k_hop_predictions = output.argmax(dim=1)[k_hop_tensor]
                k_hop_success = (k_hop_predictions == target_class).sum().item()
                
                results[f'{k}_hop'] = {
                    'success': k_hop_success,
                    'total': len(k_hop_nodes),
                    'asr': k_hop_success / len(k_hop_nodes)
                }
            else:
                results[f'{k}_hop'] = {
                    'success': 0,
                    'total': 0,
                    'asr': 0.0
                }
                
    except Exception as e:
        print(f"Warning: Error in distance calculation: {e}")

        for k in [1, 2, 3]:
            results[f'{k}_hop'] = {
                'success': 0,
                'total': 0,
                'asr': 0.0
            }
    
    return results

def calculate_accuracy_by_distance(attack_nodes, edge_index, output, original_labels, device=None, num_nodes=None):
    """
    Calculate classification accuracy for nodes at different hop distances from attack nodes.
    
    Args:
        attack_nodes: tensor of attack node indices
        edge_index: edge index tensor
        output: model predictions (log probabilities)
        original_labels: original node labels (ground truth)
        device: device to use
        num_nodes: total number of nodes in the graph
    
    Returns:
        dict: Accuracy statistics for different hop distances
    """
    if device is None:
        device = edge_index.device
    
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    valid_attack_mask = (attack_nodes >= 0) & (attack_nodes < num_nodes)
    valid_attack_nodes = attack_nodes[valid_attack_mask]
    
    if len(valid_attack_nodes) == 0:
        return {
            'attack_nodes': {'correct': 0, 'total': 0, 'accuracy': 0.0},
            '1_hop': {'correct': 0, 'total': 0, 'accuracy': 0.0},
            '2_hop': {'correct': 0, 'total': 0, 'accuracy': 0.0},
            '3_hop': {'correct': 0, 'total': 0, 'accuracy': 0.0}
        }
    
    results = {}

    original_graph_size = len(original_labels)
    valid_attack_nodes_filtered = [node.item() for node in valid_attack_nodes if node.item() < original_graph_size]
    
    if len(valid_attack_nodes_filtered) > 0:
        attack_tensor = torch.tensor(valid_attack_nodes_filtered, device=device)
        attack_predictions = output.argmax(dim=1)[attack_tensor]
        attack_labels = original_labels[attack_tensor]
        attack_correct = (attack_predictions == attack_labels).sum().item()
        results['attack_nodes'] = {
            'correct': attack_correct,
            'total': len(valid_attack_nodes_filtered),
            'accuracy': attack_correct / len(valid_attack_nodes_filtered) if len(valid_attack_nodes_filtered) > 0 else 0.0
        }
    else:
        results['attack_nodes'] = {
            'correct': 0,
            'total': 0,
            'accuracy': 0.0
        }

    try:

        adj_list = {}
        for i in range(num_nodes):
            adj_list[i] = []
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src < num_nodes and dst < num_nodes:
                adj_list[src].append(dst)
                adj_list[dst].append(src)

        distances = [-1] * num_nodes
        queue = []

        for node in valid_attack_nodes:
            node_idx = node.item()
            if node_idx < num_nodes:
                distances[node_idx] = 0
                queue.append(node_idx)

        while queue:
            current = queue.pop(0)
            current_dist = distances[current]
            
            if current_dist < 3:
                for neighbor in adj_list[current]:
                    if distances[neighbor] == -1:
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)

        for k in [1, 2, 3]:
            k_hop_nodes = [i for i in range(num_nodes) if distances[i] == k]
            k_hop_nodes = [i for i in k_hop_nodes if i < original_graph_size]
            
            if len(k_hop_nodes) > 0:
                k_hop_tensor = torch.tensor(k_hop_nodes, device=device)
                k_hop_predictions = output.argmax(dim=1)[k_hop_tensor]
                k_hop_labels = original_labels[k_hop_tensor]
                k_hop_correct = (k_hop_predictions == k_hop_labels).sum().item()
                
                results[f'{k}_hop'] = {
                    'correct': k_hop_correct,
                    'total': len(k_hop_nodes),
                    'accuracy': k_hop_correct / len(k_hop_nodes) if len(k_hop_nodes) > 0 else 0.0
                }
            else:
                results[f'{k}_hop'] = {
                    'correct': 0,
                    'total': 0,
                    'accuracy': 0.0
                }
                
    except Exception as e:
        print(f"Warning: Error in distance calculation for accuracy: {e}")
        for k in [1, 2, 3]:
            results[f'{k}_hop'] = {
                'correct': 0,
                'total': 0,
                'accuracy': 0.0
            }
    
    return results

def get_split(args,data, device):
    rs = np.random.RandomState(10)
    perm = rs.permutation(data.num_nodes)
    train_number = int(0.2*len(perm))
    idx_train = torch.tensor(sorted(perm[:train_number])).to(device)
    if args.dataset == 'Computers' or args.dataset == 'Photo':
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
    data.train_mask = torch.zeros_like(data.train_mask)
    data.train_mask[idx_train] = True

    val_number = int(0.1*len(perm))
    idx_val = torch.tensor(sorted(perm[train_number:train_number+val_number])).to(device)
    if args.dataset == 'Computers' or args.dataset == 'Photo':
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True


    test_number = int(0.1*len(perm))
    idx_test = torch.tensor(sorted(perm[train_number+val_number:train_number+val_number+test_number])).to(device)
    if args.dataset == 'Computers' or args.dataset == 'Photo':
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True

    idx_clean_test = idx_test[:int(len(idx_test)/2)]
    idx_atk = idx_test[int(len(idx_test)/2):]
    # idx_atk = idx_atk[10:20]
    return data, idx_train, idx_val, idx_clean_test, idx_atk
# %%