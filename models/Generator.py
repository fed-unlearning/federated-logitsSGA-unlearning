import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim + num_classes, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, int(np.prod(output_dim)))
        self.output_dim = output_dim

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, *self.output_dim)
        return x


def train_generator(generator, global_model, device, target_class_index, num_classes, noise_dim=100, num_steps=1000,
                    batch_size=64):
    print("Entering train_generator function.")
    generator.train()
    optimizer = optim.Adam(generator.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

    for step in range(num_steps):
        noise = torch.randn(batch_size, noise_dim).to(device)
        labels = torch.full((batch_size,), target_class_index, dtype=torch.long).to(device)
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float().to(device)

        optimizer.zero_grad()
        generated_features = generator(noise, labels_one_hot)
        logits = global_model(generated_features)

        loss = loss_func(logits, labels_one_hot)
        loss.backward()
        optimizer.step()

    print("Exiting train_generator function.")
    return generator
