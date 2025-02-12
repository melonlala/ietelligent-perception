import torch, os
from tqdm import tqdm
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from model import VAE
from visual import load_data, vis_lat, vis_img, compute_loss, test  

if torch.cuda.is_available():
    device = torch.device("cuda:6")
else:
    device = torch.device('cpu')
print('device is ', device)

batch_size = 40
lr = 1e-5
in_dim = 28 * 28
hid_dim = 2000
z_dim = 200
en_layer = 3
num_epoch = 200
i_val = num_epoch // 10
i_print = 100
beta = 0.1
train_loader, test_loader, vis_data = load_data(batch_size=batch_size)

output_path = os.path.join('exp', f"best")
writer = SummaryWriter(output_path)

model = VAE(in_dim = in_dim, hid_dim = hid_dim, lat_dim = z_dim, en_layers = en_layer).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

total_loss = []
total_loss_rec = []
total_loss_kl = []

for i in tqdm(range(num_epoch)):
    model.train()
    # args.beta = beta * min(4*(1+i) / args.epoch, 1.0)  # apply dynamic beta
    train_loss, rec_loss, kl_loss = 0, 0, 0
    for idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        y, mu, logvar2, _ = model(data)
        optimizer.zero_grad()
        loss, detail = compute_loss(data.view(-1, in_dim), y, mu, logvar2, beta)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        rec_loss += detail[0].item()
        kl_loss += detail[1].item()
        if (idx + 1) % i_print == 0:
            print(f"   Iter[{idx+1}] loss:{loss.item()}, rec loss:{detail[0].item()}, kl loss:{detail[1].item()}")
            
    total_loss.append(train_loss/len(train_loader))
    total_loss_rec.append(rec_loss/len(train_loader))
    total_loss_kl.append(kl_loss/len(train_loader))

    print(f"Epoch[{i}]: loss: {total_loss[-1]}")
    writer.add_scalar('train loss', total_loss[-1], i)
    writer.add_scalar('train rec loss', total_loss_rec[-1], i)
    writer.add_scalar('train kl loss', total_loss_kl[-1], i)
    if (i+1) % i_val == 0:
        vis_lat(i+1, output_path, model, z_dim, writer, device)
        test(i+1, output_path, test_loader, model, device, in_dim, z_dim, beta, writer)
        vis_img(i+1, output_path, vis_data.to(device), model, writer)
        model.save_checkpoint(output_path)
