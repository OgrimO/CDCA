#%%
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from models.GCN import GCN
from models.GAT import GAT
from models.SAGE import GraphSage
from models.reconstruct import MLPAE
from models.MLP import MLP

#%%
class GradWhere(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, thrd, device):

        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        

        return grad_input, None, None

class GraphTrojanNet(nn.Module):
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat,nout*nfeat)
        self.edge = nn.Linear(nfeat, int(nout*(nout-1)/2))
        self.device = device

    def forward(self, input, thrd):

        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)

        feat = self.feat(h)
        edge_weight = self.edge(h)
        # feat = GW(feat, thrd, self.device)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight

class HomoLoss(nn.Module):
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, args, device, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.args = args
        self.device = device
        self.temperature = temperature
    
    def simi(self, z1, z2):

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        s = torch.mm(z1, z2.t())
        return s
    
    def forward(self, neighbor_embeddings, target_class_embeddings, original_class_embeddings):
        f = lambda x: torch.exp(x / self.temperature)
        sim_original = f(self.simi(neighbor_embeddings, original_class_embeddings))
        sim_target = f(self.simi(neighbor_embeddings, target_class_embeddings))
        denominator = sim_original.sum(1, keepdim=True) + sim_target.sum(1, keepdim=True)
        loss = -torch.log(sim_original.sum(1, keepdim=True) / denominator).mean()
        
        return loss

class SimilarityLoss(nn.Module):
    def __init__(self, similarity_type='cosine', temperature=1.0):
        super(SimilarityLoss, self).__init__()
        self.similarity_type = similarity_type
        self.temperature = temperature
    
    def forward(self, output, target, num_classes):
        probs = torch.exp(output)
        
        target_onehot = F.one_hot(target, num_classes).float()
        
        if self.similarity_type == 'cosine':
            cosine_sim = F.cosine_similarity(probs, target_onehot, dim=1)
            loss = (1 - cosine_sim).mean()
            
        elif self.similarity_type == 'euclidean':
            euclidean_dist = torch.norm(probs - target_onehot, p=2, dim=1)
            loss = euclidean_dist.mean()
            
        elif self.similarity_type == 'kl_divergence':
            loss = F.kl_div(output, target_onehot, reduction='batchmean')
            
        elif self.similarity_type == 'js_divergence':
            m = 0.5 * (probs + target_onehot)
            kl_pm = F.kl_div(torch.log(probs + 1e-10), m, reduction='none').sum(dim=1)
            kl_qm = F.kl_div(torch.log(target_onehot + 1e-10), m, reduction='none').sum(dim=1)
            js_div = 0.5 * (kl_pm + kl_qm)
            loss = js_div.mean()
            
        elif self.similarity_type == 'dot_product':
            dot_prod = (probs * target_onehot).sum(dim=1)
            loss = -dot_prod.mean()
            
        elif self.similarity_type == 'focal':
            probs_target = probs.gather(1, target.unsqueeze(1)).squeeze(1)
            cosine_sim = F.cosine_similarity(probs, target_onehot, dim=1)
            focal_weight = (1 - cosine_sim) ** 2
            loss = -(focal_weight * torch.log(probs_target + 1e-10)).mean()
            
        else:
            loss = F.nll_loss(output, target)
        
        return loss

#%%
import numpy as np
class Backdoor:

    def __init__(self,args, device):
        self.args = args
        self.device = device
        self.weights = None
        self.trigger_index = self.get_trigger_index(args.trigger_size)
        self.use_ood_detector = getattr(self.args, 'use_ood_detector', False)
        self.idx_neighbors = None
    
    def get_trigger_index(self,trigger_size):
        edge_list = []
        edge_list.append([0,0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j,k])
        edge_index = torch.tensor(edge_list,device=self.device).long().T
        return edge_index

    def get_trojan_edge(self,start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0,0] = idx
            edges[1,0] = start
            edges[:,1:] = edges[:,1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list,dim=1)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])

        return edge_index
        
    def inject_trigger(self, idx_attach, features,edge_index,edge_weight,device):
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd) # may revise the process of generate
        
        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=device),trojan_weights],dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(device)

        update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights

    def get_k_hop_neighbors(self, idx_attach, edge_index, hops=1):
        edge_index_np = edge_index.detach().cpu().numpy()
        idx_attach_np = idx_attach.detach().cpu().numpy()
        if hops <= 0:
            return torch.tensor([], dtype=torch.long, device=self.device)

        adj_map = {}
        for src, dst in zip(edge_index_np[0], edge_index_np[1]):
            adj_map.setdefault(src, set()).add(dst)
            adj_map.setdefault(dst, set()).add(src)

        current = set(idx_attach_np)
        visited = set(idx_attach_np)
        result = set()

        for _ in range(hops):
            next_nodes = set()
            for node in current:
                for nbr in adj_map.get(node, []):
                    if nbr not in visited:
                        next_nodes.add(nbr)
                        visited.add(nbr)
            current = next_nodes
        result = visited - set(idx_attach_np)
        if len(result) == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)
        return torch.tensor(list(result), dtype=torch.long, device=self.device)

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach,idx_unlabeled, address='', test=False):

        args = self.args
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight

        self.shadow_model = GCN(
                         args=self.args,
                         nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         dropout=0.0, device=self.device).to(self.device)

        self.trojan = GraphTrojanNet(self.device, features.shape[1], args.trigger_size, layernum=2).to(self.device)
        self.homo_loss = HomoLoss(self.args, self.device)
        self.contrastive_loss = ContrastiveLoss(self.args, self.device, temperature=1.0)
        
        if self.use_ood_detector:
            print("Use OOD detector")
            self.ood_detector = MLP(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=2,
                         dropout=0.0, device=self.device).to(self.device)
            self.ood_optimizer = optim.Adam(self.ood_detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            self.ood_training_steps = getattr(args, 'ood_training_steps', 1)
            self.ood_loss_weight = getattr(args, 'ood_loss_weight', 1.0)
            features_select = features[torch.cat([idx_train, idx_attach, idx_unlabeled])]
            AE = MLPAE(features_select, features_select, self.device, args.rec_epochs)
            AE.fit()
            rec_score_ori = AE.inference(features_select)
            mean_ori = torch.mean(rec_score_ori)
            std_ori = torch.std(rec_score_ori)
            condition = torch.abs(rec_score_ori - mean_ori) < args.range * std_ori
            selected_features = features_select[condition]
        else:
            self.ood_detector = None
            self.ood_optimizer = None
            self.ood_training_steps = 0
            self.ood_loss_weight = 0.0
            selected_features = None
        
        similarity_type = getattr(args, 'similarity_loss_type', 'nll')
        temperature = getattr(args, 'similarity_temperature', 1.0)
        self.similarity_loss = SimilarityLoss(similarity_type=similarity_type, temperature=temperature)

        suppress_hops = getattr(args, 'neighbor_suppress_hops', 1)
        suppress_margin = getattr(args, 'suppress_margin', 0.1)

        self.idx_neighbors = self.get_k_hop_neighbors(idx_attach, edge_index, suppress_hops)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,args.trigger_size).to(self.device)

        poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        address = address
        
        if test==True:

            state_dict = torch.load(address)
            self.trojan.load_state_dict(state_dict)
            return 0

        loss_best = 1e8
        for i in range(args.trojan_epochs):
            self.trojan.train()
            for j in range(self.args.inner):

                optimizer_shadow.zero_grad()
                # optimizer_trigger.zero_grad()
                trojan_feat, trojan_weights = self.trojan(features[idx_attach],args.thrd)
                trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1,features.shape[1]])
                poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        
                poison_x = torch.cat([features,trojan_feat])

                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)

                idx_combined = torch.cat([idx_train, idx_attach])
                loss_inner = self.similarity_loss(output[idx_combined], self.labels[idx_combined], labels.max().item() + 1)
                
                loss_inner.backward()
                optimizer_shadow.step()
                # optimizer_trigger.step()
                if self.use_ood_detector and self.ood_detector is not None:
                    self.ood_detector.train()
                    trigger_detached = trojan_feat.detach()
                    num_trigger_nodes = trigger_detached.shape[0]
                    if len(idx_train) > 0 and num_trigger_nodes > 0:
                        for _ in range(self.ood_training_steps):

                            combined_ood = torch.cat([selected_features, trigger_detached], dim=0)
                            ood_labels = torch.cat([
                                torch.zeros(selected_features.shape[0], dtype=torch.long, device=self.device),
                                torch.ones(trigger_detached.shape[0], dtype=torch.long, device=self.device)
                            ], dim=0)
                            self.ood_optimizer.zero_grad()
                            logits_ood = self.ood_detector(combined_ood)
                            loss_detector = F.nll_loss(logits_ood, ood_labels)
                            loss_detector.backward()
                            self.ood_optimizer.step()

            optimizer_trigger.zero_grad()
            # rs = np.random.RandomState(self.args.seed)
            # idx_outter = torch.cat([idx_attach,idx_unlabeled[rs.choice(len(idx_unlabeled),size=512,replace=False)]])
            # idx_outter = idx_unlabeled[rs.choice(len(idx_unlabeled),size=512,replace=False)]
            # idx_outter = idx_outter[~torch.isin(idx_unlabeled, idx_attach)]


            trojan_feat, trojan_weights = self.trojan(features[idx_attach], args.thrd)
            trojan_weights = torch.cat([torch.ones([len(trojan_feat), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)
            trojan_weights_flat = trojan_weights.flatten()
            trojan_feat = trojan_feat.view([-1, features.shape[1]])

            trojan_edge = self.get_trojan_edge(len(features), idx_attach, args.trigger_size).to(self.device)
            update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)
            trigger_edge_weights = torch.cat([trojan_weights_flat, trojan_weights_flat])
            update_edge_weights = torch.cat([edge_weight, trojan_weights_flat, trojan_weights_flat])
            update_feat = torch.cat([features, trojan_feat])
            output_attach = self.shadow_model(update_feat, update_edge_index, update_edge_weights)            

            labels_target = labels.clone()
            labels_target[idx_attach] = args.target_class
            idx_combined = torch.cat([idx_train, idx_attach])
            loss_target = self.similarity_loss(output_attach[idx_combined], labels_target[idx_combined], labels.max().item() + 1)


            loss_neighbor = torch.tensor(0.0, device=self.device)
            loss_2hop_suppress = torch.tensor(0.0, device=self.device)
            neighbor_clean_acc = None
            
            if self.idx_neighbors is not None and self.idx_neighbors.numel() > 0:

                all_embeddings = self.shadow_model.get_h(update_feat, update_edge_index)
                neighbor_embeddings = all_embeddings[self.idx_neighbors]

                with torch.no_grad():
                    clean_output = self.shadow_model(features, edge_index, edge_weight)
                    pseudo_labels = clean_output.argmax(dim=1)

                    target_class_mask = labels[idx_train] == args.target_class
                    if target_class_mask.sum().item() > 0:
                        target_class_nodes = idx_train[target_class_mask]
                        target_class_embeddings = all_embeddings[target_class_nodes].detach()
                    else:
                        target_class_embeddings = None

                    neighbor_pseudo_labels = pseudo_labels[self.idx_neighbors]
                    unique_labels = neighbor_pseudo_labels.unique()
                    original_class_embeddings_list = []
                    for label in unique_labels:
                        label_value = int(label.item())
                        if label_value == args.target_class:
                            continue
                        label_mask = pseudo_labels == label_value
                        if label_mask.sum().item() == 0:
                            continue
                        label_nodes = label_mask.nonzero(as_tuple=True)[0]
                        label_embeddings = all_embeddings[label_nodes].detach()
                        original_class_embeddings_list.append(label_embeddings)
                    
                    if len(original_class_embeddings_list) > 0:
                        original_class_embeddings = torch.cat(original_class_embeddings_list, dim=0)
                    else:
                        original_class_embeddings = None

                if target_class_embeddings is not None and original_class_embeddings is not None:
                    loss_neighbor = self.contrastive_loss(
                        neighbor_embeddings,
                        target_class_embeddings,
                        original_class_embeddings
                    )
                
                probs_neighbors = F.softmax(output_attach[self.idx_neighbors], dim=1)
                target_class_prob = probs_neighbors[:, args.target_class]
                loss_2hop_suppress = torch.relu(target_class_prob - suppress_margin).mean()

                with torch.no_grad():
                    preds_neighbors = output_attach[self.idx_neighbors].argmax(dim=1)
                    neighbor_clean_acc = (preds_neighbors == labels[self.idx_neighbors]).float().mean().item()


            loss_target_weight = getattr(args, 'loss_target_weight', 1.0)

            loss_neighbor_weight = getattr(args, 'loss_2hop_weight', 1.0)
            loss_2hop_suppress_weight = getattr(args, 'loss_2hop_suppress_weight', 5.0)
            
            loss_total = (loss_target_weight * loss_target +
                          loss_neighbor_weight * loss_neighbor+
                        #   homo_weight * loss_homo+
                          loss_2hop_suppress_weight * loss_2hop_suppress)
            # print("loss_homo: ", loss_homo)
            # print("loss_neighbor: ", loss_neighbor)
            # if self.use_ood_detector:
            #     loss_total = loss_total + self.ood_loss_weight * loss_ood
            
            if loss_total < loss_best:
                loss_best = float(loss_total)
                self.weights = deepcopy(self.trojan.state_dict())

            loss_total.backward()
            optimizer_trigger.step()

            acc_train_clean = utils.accuracy(output[idx_train], labels[idx_train])

            target_mask = labels[idx_attach] != args.target_class
            if target_mask.sum() > 0:
                target_asr = (output[idx_attach[target_mask]].argmax(dim=1) == args.target_class).float().mean().item()
            else:
                target_asr = (output[idx_attach].argmax(dim=1) == args.target_class).float().mean().item()
            
            neighbor_asr = 0.0
            if self.idx_neighbors is not None and self.idx_neighbors.numel() > 0:

                neighbor_mask = labels[self.idx_neighbors] != args.target_class

                original_num_nodes = len(features)
                neighbor_mask = neighbor_mask & (self.idx_neighbors < original_num_nodes)
                if neighbor_mask.sum() > 0:
                    neighbor_asr = (output[self.idx_neighbors[neighbor_mask]].argmax(dim=1) == args.target_class).float().mean().item()

            if ((i + 1) % 50 == 0 or i == args.trojan_epochs - 1):
                print(f"[CDCA][Epoch {i+1}/{args.trojan_epochs}] Target_ASR: {target_asr:.4f}, Neighbor_ASR: {neighbor_asr:.4f}, Clean_Acc: {acc_train_clean:.4f}")
                if neighbor_clean_acc is not None:
                    print(f"  Neighbor clean-label ACC: {neighbor_clean_acc:.4f}")
                if loss_neighbor.item() > 0:
                    print(f"  loss_neighbor (contrastive): {loss_neighbor.item():.6f}, loss_2hop_suppress: {loss_2hop_suppress.item():.6f}")
        print("loss_best: ", loss_best)
        
        self.trojan.load_state_dict(self.weights)

        self.trojan.eval()

    def get_poisoned(self):

        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_attach,self.features,self.edge_index,self.edge_weights,self.device)
        poison_labels = self.labels
        poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels

# %%
