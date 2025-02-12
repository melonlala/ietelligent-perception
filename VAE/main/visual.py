import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np

# visualize latent space (only support lat_dim = 1 or 2)
def vis_lat(i, path, model, lat_dim, writer, device):
    if lat_dim > 2:
        return
    if not os.path.exists(f'./{path}/images/{i}'):
        os.makedirs(f'./{path}/images/{i}', exist_ok=True)
    if lat_dim == 1:
        z = torch.linspace(-5.0, 5.0, 100).to(device).unsqueeze(-1)
        y = model.generate(z).view(10, 10, 28, 28).permute(0, 2, 1, 3).reshape(10 * 28, 10 * 28).unsqueeze(0)
        save_image(y, f'./{path}/images/{i}/lat' + '.png')
        writer.add_image("generate_lat", y, i)
    else:
        z = torch.linspace(-5.0, 5.0, 10).to(device)
        x, y = torch.meshgrid(z, z)
        out = model.generate(torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1).reshape(-1, 2))
        out = out.view(10, 10, 28, 28).permute(0, 2, 1, 3).reshape(10 * 28, 10 * 28).unsqueeze(0)
        save_image(out, f'./{path}/images/{i}/lat' + '.png')
        writer.add_image("generate_lat", out, i)

# map dataset to latent space (only support lat_dim = 1 or 2 or 3)
def test(i, path, data_loader, model, device, in_dim, lat_dim, beta, writer):
    model.eval()
    test_loss = 0
    lat_x, lat_y = [], []
    for idx, (data, label) in enumerate(data_loader):
        data = data.to(device)
        y, mu, logvar2, z = model(data)
        loss, detail = compute_loss(data.view(-1, in_dim), y, mu, logvar2, beta)
        test_loss += detail[0].item()  # only rec_loss
        lat_x.append(z)
        lat_y.append(label)
    writer.add_scalar('test loss', test_loss/len(data_loader), i)
    lat_x = torch.stack(lat_x).reshape(-1, lat_dim).cpu()
    lat_y = torch.stack(lat_y).reshape(-1, 1)
    lat = torch.cat([lat_x, lat_y], dim=-1).detach().numpy()
    fig = plt.figure()
    if lat_dim <= 2:
        ax = fig.add_subplot()
        for num in range(10):
            x = np.array([t for t in lat if t[-1]==num])
            ax.scatter(x[:, 0], x[:, 1], label=str(num))
    elif lat_dim == 3:
        ax = fig.add_subplot(projection='3d')
        for num in range(10):
            x = np.array([t for t in lat if t[-1]==num])
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], label=str(num))
    else:
        return

    ax.legend()
    if not os.path.exists(f'./{path}/images/{i}'):
        os.makedirs(f'./{path}/images/{i}', exist_ok=True)
    save_image(y, f'./{path}/images/{i}/vis_data' + '.png')
    writer.add_figure('vis_data', fig, i)


def vis_img(i, path, data, model, writer=None):
    model.eval()
    y, mu, logvar2, _ = model(data)
    nelem = data.size(0)
    nrow = 10
    if nelem % nrow != 0:
        print("nelem/nrow is not an integer!")
        exit(0)
    data = data.view(nelem//nrow, nrow, 28, 28)
    y = y.view(data.shape)
    data = data.permute(0, 2, 1, 3).reshape(nelem//nrow*28, -1)
    y = y.permute(0,2,1,3).reshape(nelem//nrow*28, -1)
    img = torch.cat([data, y]).unsqueeze(0)

    if not os.path.exists(f'./{path}/images/{i}'):
        os.makedirs(f'./{path}/images/{i}', exist_ok=True)
    save_image(img, f'./{path}/images/{i}/vis_img' + '.png')

    writer.add_image('vis_img', img, i)


def compute_loss(true, pred, mu, logvar2, beta):
    loss_recon = F.binary_cross_entropy(pred, true, reduction='sum')
    loss_KL = -0.5 * torch.sum(1 + logvar2 - mu.pow(2) - logvar2.exp())
    return loss_recon + beta * loss_KL, [loss_recon, loss_KL]


def load_data(batch_size):
    # Download MNIST Dataset
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

    # MNist Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train data num: {len(train_loader)}, test data num: {len(test_loader)}")
    
    for data, _ in test_loader:
        vis_data = data
        break
    return train_loader, test_loader, vis_data

def add_noise(x, mean=0., std=1e-6):
    # [B, 1, 28, 28]
    rand = torch.rand([x.shape[0], 1, 1, 1])
    rand = torch.where(rand > 0.5, 1., 0.).to(x.device)
    white_noise = torch.normal(mean, std, size=x.shape, device=x.device)
    noise_x = x + white_noise * rand  # add noise
    noise_x = torch.clip(noise_x, 0., 1.)  # clip to [0, 1]
    return noise_x