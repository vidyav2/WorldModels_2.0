import argparse
from multiprocessing import reduction
from os.path import join, exists, isdir
from os import mkdir, listdir

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from models.vae import VAE
from utils.misc import save_checkpoint, LSIZE, RED_SIZE
from utils.learning import EarlyStopping, ReduceLROnPlateau
from data.loaders import RolloutObservationDataset

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', type=str, required=True, help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')

args = parser.parse_args()
cuda = torch.cuda.is_available()

torch.manual_seed(123)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

dataset_path = 'datasets/carracing'
if not exists(dataset_path):
    raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
print(f"Contents of {dataset_path}: {listdir(dataset_path)}")

thread_dirs = [join(dataset_path, d) for d in listdir(dataset_path) if 'thread' in d and isdir(join(dataset_path, d))]
if not thread_dirs:
    raise ValueError("No thread directories found in the dataset path.")

print(f"Found thread directories: {thread_dirs}")

dataset_train = RolloutObservationDataset(thread_dirs, transform_train, train=True)
dataset_test = RolloutObservationDataset(thread_dirs, transform_test, train=False)

if len(dataset_train) == 0 or len(dataset_test) == 0:
    raise ValueError("Datasets are empty. Check if the data is correctly placed in the specified path.")

train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

model = VAE(3, LSIZE).to(device)
optimizer = optim.Adam(model.parameters()) #Added learning rate
#optimizer = optim.Adam(model.parameters(), lr=1e-4) #Added learning rate
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30) #Increased patience from 30 to 50

def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    #BCE = F.mse_loss(recon_x, x, size_average=False)
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD

def convert_state_dict(state_dict):
    """ Convert state dict to new format with NoisyLinear layers """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if 'fc_mu.weight' in k or 'fc_logsigma.weight' in k or 'fc1.weight' in k:
            new_key = k.replace('weight', 'weight_mu')
        if 'fc_mu.bias' in k or 'fc_logsigma.bias' in k or 'fc1.bias' in k:
            new_key = k.replace('bias', 'bias_mu')
        new_state_dict[new_key] = v

    # Initialize the new keys for sigma and epsilon with appropriate values
    keys = list(new_state_dict.keys())
    for key in keys:
        if 'weight_mu' in key:
            base_key_sigma = key.replace('weight_mu', 'weight_sigma')
            base_key_epsilon = key.replace('weight_mu', 'weight_epsilon')
            new_state_dict[base_key_sigma] = torch.zeros_like(new_state_dict[key])
            new_state_dict[base_key_epsilon] = torch.zeros_like(new_state_dict[key])
        if 'bias_mu' in key:
            base_key_sigma = key.replace('bias_mu', 'bias_sigma')
            base_key_epsilon = key.replace('bias_mu', 'bias_epsilon')
            new_state_dict[base_key_sigma] = torch.zeros_like(new_state_dict[key])
            new_state_dict[base_key_epsilon] = torch.zeros_like(new_state_dict[key])

    return new_state_dict

def train(epoch):
    """ One training epoch """
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        model.reset_noise()  # Reset noise in NoisyLinear layers
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)
            #))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

def test():
    """ One test epoch """
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            model.reset_noise()  # Reset noise in NoisyLinear layers
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

vae_dir = join(args.logdir, 'vaeNew')
if not exists(vae_dir):
    mkdir(vae_dir)
    mkdir(join(vae_dir, 'samplesNew'))

reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    # Convert state dict to match new format
    new_state_dict = convert_state_dict(state['state_dict'])
    model.load_state_dict(new_state_dict, strict=False)

    # Reinitialize the optimizer with the model's parameters
    optimizer = optim.Adam(model.parameters())
    try:
        optimizer.load_state_dict(state['optimizer'])
    except ValueError:
        print("Optimizer state dict mismatch, starting with a fresh optimizer.")

    scheduler.load_state_dict(state['scheduler'])
    if 'earlystopping' in state:
        earlystopping.load_state_dict(state['earlystopping'])
    else:
        print("Early stopping state dict not found, starting with a fresh early stopping.")

cur_best = None

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_loss = test()
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    best_filename = join(vae_dir, 'best.tar')
    filename = join(vae_dir, 'checkpoint.tar')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, is_best, filename, best_filename)

    if not args.nosamples:
        with torch.no_grad():
            sample = torch.randn(RED_SIZE, LSIZE).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample.view(64, 3, RED_SIZE, RED_SIZE),
                       join(vae_dir, 'samplesNew/sample_' + str(epoch) + '_a.png'))

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break
