#!/usr/bin/env python
# coding: utf-8
###############################################################################################
## Adapted from SPEAR: A Structure-Preserving Manipulation Method for Graph Backdoor Attacks ##
###############################################################################################
# In[1]: 

import warnings
# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak.*")

import imp
import time
import argparse
from copy import deepcopy
from unittest import result
import numpy as np
import torch
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI
from help_funcs import reconstruct_prune_unrelated_edge

# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import torch.nn as nn
from ogb.nodeproppred import PygNodePropPredDataset


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','ogbn-arxiv'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)

parser.add_argument('--target_class', type=int, default=0)

parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')

parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--vs_number', type=int, default=40,
                    help="ratio of poisoning nodes relative to the full graph")

parser.add_argument('--layer', type=int, default=2,
                    help='Number of layers.')

parser.add_argument('--dis_weight', type=float, default=0.5,
                    help="Weight of cluster distance")

parser.add_argument('--selection_method', type=str, default='none',
                    choices=['loss','conf','cluster','none','cluster_degree','target_cluster_distance'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')

parser.add_argument('--neighbor_suppress_hops', type=int, default=1,
                    help="suppress hops")

parser.add_argument('--loss_target_weight', type=float, default=1,
                    help="Weight of target loss term")
# parser.add_argument('--homo_loss_weight', type=float, default=50,
#                     help="Weight of optimize similarity loss")
parser.add_argument('--similarity_loss_type', type=str, default='nll',
                    choices=['nll', 'cosine', 'euclidean', 'kl_divergence', 'js_divergence', 'dot_product', 'focal'],
                    help="Type of similarity loss to use (nll=cross-entropy, cosine=cosine similarity, etc.)")
parser.add_argument('--similarity_temperature', type=float, default=1.0,
                    help="Temperature parameter for similarity loss")
parser.add_argument('--loss_2hop_weight', type=float, default=2,
                    help="Weight of 2-hop neighbor classification loss")
parser.add_argument('--loss_2hop_suppress_weight', type=float, default=1,
                    help="Weight of 2-hop neighbor suppression loss")

parser.add_argument('--use_ood_detector', type=bool,default=False,
                    help="Enable OOD detector regularization during trigger training")
parser.add_argument('--ood_hidden', type=int, default=64,
                    help="Hidden units for OOD detector MLP")
parser.add_argument('--ood_lr', type=float, default=0.005,
                    help="Learning rate for OOD detector")
parser.add_argument('--ood_training_steps', type=int, default=100,
                    help="Number of OOD detector update steps per inner iteration")
parser.add_argument('--ood_loss_weight', type=float, default=1.0,
                    help="Weight for OOD detector loss term")
parser.add_argument('--range', type=float, default=0.01,
                    help="ratio of poisoning nodes relative to the full graph")
  

parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN','ABL'],
                    help='Model used to attack')

parser.add_argument('--defense_mode', type=str, default="none",
                    choices=['prune', 'isolate', 'none', 'reconstruct'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.1,
                    help="Threshold of prunning edges")

parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")
parser.add_argument('--rec_epochs', type=int,  default=100,
                    help='Number of epochs to train benign and backdoor model.')

parser.add_argument('--trigger_generator_address', type=str, default='./weights/GTA/Cora/GTA_Cora_weights.pth')
parser.add_argument('--pre_train_param', type=str, default='./weights/GTA/Cora/GTA_Cora.pt')

parser.add_argument('--attack', type=str, default='GTA',
                    help="attack type")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args) 

#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/', \
                        name=args.dataset,\
                        transform=transform)
elif(args.dataset == 'ogbn-arxiv'):

    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split()

data = dataset[0].to(device)

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)

#%% 
from utils import get_split, calculate_asr_by_distance, calculate_accuracy_by_distance
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)

from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]


# In[9]:
from models.CDCA import Backdoor
import heuristic_selection as hs

unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
size = args.vs_number 
print("#Attach Nodes:{}".format(size))

from models.construct import model_construct
if(args.selection_method == 'none'):
    idx_attach = hs.obtain_attach_nodes(args,unlabeled_idx,size)
elif(args.selection_method == 'cluster'):
    idx_attach = hs.cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif(args.selection_method == 'cluster_degree'):
    idx_attach = hs.cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif(args.selection_method == 'target_cluster_distance'):
    idx_attach = hs.target_cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)

mask = data.y[idx_attach] != args.target_class
mask = mask.to(device)
idx_attach = idx_attach[(data.y[idx_attach] != args.target_class).nonzero().flatten()]
print("idx_attach: {}".format(idx_attach))
print("num_attach: ", len(idx_attach))
unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)

known_nodes = torch.cat([idx_train,idx_attach]).to(device)

edge_weight = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)



model = Backdoor(args,device)
model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
# torch.save(model.trojan.state_dict(), args.trigger_generator_address)

# model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_attach, unlabeled_idx, args.trigger_generator_address, True)
poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()

if(args.defense_mode == 'prune'):
    poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
elif(args.defense_mode == 'reconstruct'):
    poison_edge_index,poison_edge_weights = reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,data.x,data.edge_index,device, idx_attach, large_graph=True)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
elif(args.defense_mode == 'isolate'):
    poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
    bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
else:
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
print("precent of left attach nodes: {:.3f}"\
    .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach)))
    #%%

test_model = model_construct(args,args.test_model,data,device,layer=args.layer).to(device) 

if test_model == 'ABL':
    print("Train ABL model")
    test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False, num_attach=len(idx_attach))
else:
    test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)

test_model.eval()

output = test_model(poison_x, poison_edge_index, poison_edge_weights)
induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])

# original_num_nodes = data.num_nodes
# true_target_cnt = (data.y[:original_num_nodes] == args.target_class).sum().item()

# output_full_pre = test_model(poison_x, induct_edge_index, induct_edge_weights)
# pred_target_cnt_pre = (output_full_pre.argmax(dim=1)[:original_num_nodes] == args.target_class).sum().item()
# print("\n****Target-class Count (Full Graph, pre-attack)****")
# print(f"GT target-class nodes (y=={args.target_class}) in original graph: {true_target_cnt}/{original_num_nodes}")
# print(f"Predicted as target (pre-attack): {pred_target_cnt_pre}/{original_num_nodes}")

train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
print("target class rate on Vs: {:.4f}".format(train_attach_rate))
clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

print("accuracy on clean test nodes: {:.4f}".format(clean_acc))


induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()

flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
print("\n****Attack Nodes****")
print(f"Attack nodes provided (idx_atk): {len(idx_atk)}")
print(f"Attack nodes excluding original target-class (flip_idx_atk): {len(flip_idx_atk)}")
# output = test_model(induct_x,induct_edge_index,induct_edge_weights)
ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
print("ca: {:.4f}".format(ca))
train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
print("target class rate on Vs: {:.4f}".format(train_attach_rate))
if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
    induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)

output = test_model(induct_x,induct_edge_index,induct_edge_weights)

flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
# Post-attack (after injecting triggers to idx_atk; after optional pruning)
# pred_target_cnt_post = (output.argmax(dim=1)[:original_num_nodes] == args.target_class).sum().item()
# print("\n****Target-class Count (Full Graph, post-attack)****")
# print(f"Predicted as target (post-attack): {pred_target_cnt_post}/{original_num_nodes}")
# print(f"Delta predicted-as-target: {abs(pred_target_cnt_post - true_target_cnt-len(flip_idx_atk)):+d}")

print("****After GTA Attack****")
print("ASR: {:.6f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
print("Clean Accuracy: {:.6f}".format(clean_acc))

# print("****ASR Analysis for Attack Nodes and Neighbors****")
# print(f"Total attack nodes: {len(idx_atk)}, Graph has {data.num_nodes} nodes")
# print(f"Attack node range: [{idx_atk.min().item()}, {idx_atk.max().item()}]")

asr_results_filtered = calculate_asr_by_distance(
    attack_nodes=flip_idx_atk,  # Filtered attack nodes
    edge_index=induct_edge_index,
    output=output,
    target_class=args.target_class,
    original_labels=data.y,
    device=device,
    num_nodes=data.num_nodes
)


print("flip_ASR (filtered attack nodes): {:.6f} ({}/{} nodes)".format(
    asr_results_filtered['attack_nodes']['asr'], 
    asr_results_filtered['attack_nodes']['success'], 
    asr_results_filtered['attack_nodes']['total']
))

for k in [1, 2, 3]:
    hop_key = f'{k}_hop'

    print("flip_ASR for {}-hop neighbors (filtered): {:.6f} ({}/{} nodes)".format(
        k,
        asr_results_filtered[hop_key]['asr'],
        asr_results_filtered[hop_key]['success'],
        asr_results_filtered[hop_key]['total']
    ))

accuracy_results_filtered = calculate_accuracy_by_distance(
    attack_nodes=flip_idx_atk,
    edge_index=induct_edge_index,
    output=output,
    original_labels=data.y,
    device=device,
    num_nodes=data.num_nodes
)

print("Accuracy (filtered attack nodes): {:.6f} ({}/{} nodes)".format(
    accuracy_results_filtered['attack_nodes']['accuracy'],
    accuracy_results_filtered['attack_nodes']['correct'],
    accuracy_results_filtered['attack_nodes']['total']
))

# Show neighbor accuracy for attack nodes
for k in [1, 2, 3]:
    hop_key = f'{k}_hop'

    print("Accuracy for {}-hop neighbors (filtered): {:.6f} ({}/{} nodes)".format(
        k,
        accuracy_results_filtered[hop_key]['accuracy'],
        accuracy_results_filtered[hop_key]['correct'],
        accuracy_results_filtered[hop_key]['total']
    ))

induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
output = output.cpu()
