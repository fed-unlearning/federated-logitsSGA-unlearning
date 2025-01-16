import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNSVHN, ResNetMnist, ResNetCifar,ResNetSVHN
from models.Test import test_img
from models.Aggregation import LocalUpdatePostForget, adaptive_fedavg
from models.Fed import FedAvg
from torch.utils.data import DataLoader, Subset
from models.Generator import Generator, train_generator

TARGET_CLASS_INDEX = 3
NOISE_DIM = 100

def custom_loss(outputs, labels):
    log_probs = F.log_softmax(outputs, dim=1)
    per_item_loss = F.nll_loss(log_probs, labels, reduction='none')
    base_loss = per_item_loss.mean()

    # Maximizing loss for the target class by gradient ascent
    target_mask = labels == TARGET_CLASS_INDEX
    if target_mask.any():
        target_loss = per_item_loss[target_mask].mean()
        base_loss -= 0.55 * target_loss

    # Minimizing loss for non-target classes
    non_target_mask = labels != TARGET_CLASS_INDEX
    if non_target_mask.any():
        non_target_loss = per_item_loss[non_target_mask].mean()
        base_loss += non_target_loss

    return base_loss
def forget_with_generated_data(global_model, generator, device, TARGET_CLASS_INDEX, num_classes,
                               noise_dim=100, num_steps=1000, batch_size=64):
    global_model.train()
    optimizer = optim.SGD(global_model.parameters(), lr=0.001, momentum=0.9)

    for step in range(num_steps):
        noise = torch.randn(batch_size, noise_dim).to(device)

        # Generate data for the target category
        labels_target = torch.full((batch_size // 2,), TARGET_CLASS_INDEX, dtype=torch.long).to(device)
        labels_target_one_hot = F.one_hot(labels_target, num_classes=num_classes).float().to(device)
        generated_features_target = generator(noise[:batch_size // 2], labels_target_one_hot)

        # Generate data for the non-target category
        non_target_labels = torch.randint(0, num_classes, (batch_size // 2,)).to(device)
        non_target_labels = non_target_labels[non_target_labels != TARGET_CLASS_INDEX][:batch_size // 2]
        noise_non_target = torch.randn(len(non_target_labels), noise_dim).to(device)
        non_target_labels_one_hot = F.one_hot(non_target_labels, num_classes=num_classes).float().to(device)
        generated_features_non_target = generator(noise_non_target, non_target_labels_one_hot)

        # Merge the generated data and labels
        generated_features = torch.cat([generated_features_target, generated_features_non_target], dim=0)
        labels = torch.cat([labels_target, non_target_labels], dim=0)

        optimizer.zero_grad()
        outputs = global_model(generated_features)
        loss = custom_loss(outputs, labels)
        loss.backward()
        optimizer.step()

    print('Finished forgetting operation on the target class')


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        dict_users = mnist_iid(dataset_train, args.num_users) if args.iid else mnist_noniid(dataset_train, args.num_users)
        num_classes = 10
        output_dim = (1, 28, 28)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        dict_users = cifar_iid(dataset_train, args.num_users)
        num_classes = 10
        output_dim = (3, 32, 32)

        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')

    if args.dataset == 'mnist':
        if args.model == 'cnn':
            net_glob = CNNMnist(args=args).to(args.device)
        elif args.model == 'resnet':
            net_glob = ResNetMnist(num_classes=args.num_classes).to(args.device)
        else:
            raise ValueError("Unrecognized model type for MNIST")
    elif args.dataset == 'cifar':
        if args.model == 'cnn':
            net_glob = CNNCifar(args=args).to(args.device)
        elif args.model == 'resnet':
            net_glob = ResNetCifar(num_classes=args.num_classes).to(args.device)
        else:
            raise ValueError("Unrecognized model type for CIFAR")
    elif args.dataset == 'svhn':
        if args.model == 'cnn':
            net_glob = CNNSVHN(args=args).to(args.device)
        elif args.model == 'resnet':
            net_glob = ResNetSVHN(num_classes=args.num_classes).to(args.device)
        else:
            raise ValueError("Unrecognized model type for SVHN")
    else:
        raise ValueError("Unrecognized dataset")

    # train the generator
    generator = Generator(noise_dim=NOISE_DIM, num_classes=num_classes, output_dim=output_dim).to(args.device)
    generator = train_generator(generator, net_glob, args.device, TARGET_CLASS_INDEX, num_classes=num_classes,
                                noise_dim=NOISE_DIM, num_steps=1000, batch_size=64)

    print("Dataset:", args.dataset)
    print("Model:", args.model)

    batch_size = 32
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    net_glob = (
        CNNCifar(args=args).to(args.device) if args.model == 'cnn' and args.dataset == 'cifar' else
        CNNMnist(args=args).to(args.device) if args.model == 'cnn' and args.dataset == 'mnist' else
        CNNSVHN(args=args).to(args.device) if args.model == 'cnn' and args.dataset == 'svhn' else
        ResNetCifar(num_classes=args.num_classes).to(
            args.device) if args.model == 'resnet' and args.dataset == 'cifar' else
        ResNetMnist(num_classes=args.num_classes).to(
            args.device) if args.model == 'resnet' and args.dataset == 'mnist' else
        ResNetSVHN(num_classes=args.num_classes).to(
            args.device) if args.model == 'resnet' and args.dataset == 'svhn' else
        MLP(dim_in=np.prod(dataset_train[0][0].shape), dim_hidden=200, dim_out=num_classes).to(
            args.device) if args.model == 'mlp' else
        exit('Error: unrecognized model')
    )
    print(net_glob)
    net_glob.train()
    w_glob = net_glob.state_dict()

    # federal learning
    for iter in range(args.epochs):
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_locals = []
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, sum(loss_locals) / len(loss_locals)))

    net_glob.eval()
    acc_train_target, acc_train_non_target = test_img(net_glob, train_loader, args, TARGET_CLASS_INDEX)
    acc_test_target, acc_test_non_target = test_img(net_glob, test_loader, args, TARGET_CLASS_INDEX)
    print("Training accuracy for target class: {:.2f}".format(acc_train_target))
    print("Training accuracy for non-target classes: {:.2f}".format(acc_train_non_target))
    print("Testing accuracy for target class: {:.2f}".format(acc_test_target))
    print("Testing accuracy for non-target classes: {:.2f}".format(acc_test_non_target))

    # forgetting operation
    forget_with_generated_data(net_glob, generator, args.device, TARGET_CLASS_INDEX, num_classes=num_classes,
                               noise_dim=NOISE_DIM, num_steps=1000, batch_size=64)

    # testing
    net_glob.eval()
    acc_train_target, acc_train_non_target = test_img(net_glob, train_loader, args, TARGET_CLASS_INDEX)
    acc_test_target, acc_test_non_target = test_img(net_glob, test_loader, args, TARGET_CLASS_INDEX)
    print("Training accuracy for target class: {:.2f}".format(acc_train_target))
    print("Training accuracy for non-target classes: {:.2f}".format(acc_train_non_target))
    print("Testing accuracy for target class: {:.2f}".format(acc_test_target))
    print("Testing accuracy for non-target classes: {:.2f}".format(acc_test_non_target))


    net_glob.train()
    w_glob = net_glob.state_dict()

    # aggregation
    for iter in range(10):
        w_locals = []
        loss_locals = []
        for idx in dict_users:
            local = LocalUpdatePostForget(args, dataset_train, dict_users[idx], TARGET_CLASS_INDEX)
            w, loss = local.train(copy.deepcopy(net_glob))
            w_locals.append(w)
            loss_locals.append(loss)
        w_glob = FedAvg(w_locals)
        #w_glob = adaptive_fedavg(w_glob, w_locals)
        net_glob.load_state_dict(w_glob)

        net_glob.eval()
        _, acc_train_non_target = test_img(net_glob, train_loader, args, TARGET_CLASS_INDEX)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, sum(loss_locals) / len(loss_locals)))
        print("Round {:3d}, Training accuracy for non-target classes: {:.2f}".format(iter, acc_train_non_target))


    # final test
    net_glob.eval()
    acc_train_target, acc_train_non_target = test_img(net_glob, train_loader, args, TARGET_CLASS_INDEX)
    acc_test_target, acc_test_non_target = test_img(net_glob, test_loader, args, TARGET_CLASS_INDEX)
    print("Post-recovery training accuracy for target class: {:.2f}".format(acc_train_target))
    print("Post-recovery training accuracy for non-target classes: {:.2f}".format(acc_train_non_target))
    print("Post-recovery testing accuracy for target class: {:.2f}".format(acc_test_target))
    print("Post-recovery testing accuracy for non-target classes: {:.2f}".format(acc_test_non_target))
