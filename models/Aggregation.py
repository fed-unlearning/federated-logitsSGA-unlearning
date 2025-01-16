# models/Aggregation.py
# ParamMonitor/NodeIdentifier/WeightAdjuster

import torch
import copy
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

class LocalUpdatePostForget:
    def __init__(self, args, dataset, idxs, target_class_index):
        self.args = args
        self.dataset = dataset
        self.idxs = list(idxs)
        self.target_class_index = target_class_index

    def train(self, net):
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=self.args.momentum)
        train_loader = DataLoader(Subset(self.dataset, self.idxs),
                                  batch_size=self.args.local_bs, shuffle=True)
        loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(self.args.device), labels.to(self.args.device)
            mask = labels != self.target_class_index
            data = data[mask]
            labels = labels[mask]

            if len(labels) > 0:
                optimizer.zero_grad()
                output = net(data)
                loss = F.cross_entropy(output, labels)
                loss.backward()

                # # cifar + resnet
                # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

                optimizer.step()

        return net.state_dict(), loss


def adaptive_fedavg(w_globals, w_locals):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 选择设备

    w_globals = {k: v.to(device).float() for k, v in w_globals.items()}

    w_locals = [{k: v.to(device).float() for k, v in w_local.items()} for w_local in w_locals]

    total_weight = 0.0
    # weights = []
    aggregated_params = {k: torch.zeros_like(v).float() for k, v in w_globals.items()}

    for w_local in w_locals:
        model_diff = sum(torch.norm(w_local[k].float() - w_globals[k].float(), p=2).item() for k in w_globals)
        model_diff_tensor = torch.tensor(model_diff, device=device)

        # mnist+svhn
        weight = torch.exp(-model_diff_tensor)

        # # cifar + resnet
        # weight = torch.exp(-model_diff_tensor / 10)


        total_weight += weight.item()

        # adjust the weight parameters
        for k in w_globals:
            aggregated_params[k] += w_local[k] * weight



    # normalization
    for k in aggregated_params:
        aggregated_params[k] /= total_weight

    return aggregated_params