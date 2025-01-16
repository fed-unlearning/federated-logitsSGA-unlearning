import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def test_img(net, dataloader, args, target_class):
    net.eval()
    correct_target = 0
    total_target = 0
    correct_non_target = 0
    total_non_target = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            correct_target += (predicted[labels == target_class] == labels[labels == target_class]).sum().item()
            total_target += (labels == target_class).sum().item()
            correct_non_target += (predicted[labels != target_class] == labels[labels != target_class]).sum().item()
            total_non_target += (labels != target_class).sum().item()

    acc_target = 100.0 * correct_target / total_target if total_target > 0 else 0
    acc_non_target = 100.0 * correct_non_target / total_non_target if total_non_target > 0 else 0
    return acc_target, acc_non_target