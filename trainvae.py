import argparse
from os.path import join, exists
from os import mkdir
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
import numpy as np
import glob

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
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

#root_dir = 'datasets/carracing'

#train_datasets = []
#test_datasets = []

thread_dirs = glob.glob('datasets/carracing/thread_*')

train_dataset = RolloutObservationDataset(thread_dirs, transform_train, train=True)
test_dataset = RolloutObservationDataset(thread_dirs, transform_test, train=False)
# Print the length of the datasets
print(f"Length of training dataset: {len(train_dataset)}")
print(f"Length of testing dataset: {len(test_dataset)}")

# Replace 'some_file.npz' with an actual file name from your dataset
#with np.load(thread_dirs) as data:
 #   print(list(data.keys()))  # Check for expected keys like 'observations'
  #  print(data['observations'].shape)  # This should not be empty

"""for thread_id in range(1):  # Assuming threads range from 0 to 7
    thread_dir = join(root_dir, f'thread_{thread_id}')
    print(f"Checking data in: {thread_dir}")  # Keep this for debugging
    train_dataset = RolloutObservationDataset('thread_dir', transform_train, train=True)
    test_dataset = RolloutObservationDataset(thread_dir, transform_test, train=True)

    print(f"Thread {thread_id} - Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")  # Debugging

    if len(train_dataset) >= 0:
        train_datasets.append(train_dataset)
        print(f"Added {len(train_dataset)} training samples from {thread_dir}")
    if len(test_dataset) > 0:
        test_datasets.append(test_dataset)
        print(f"Added {len(test_dataset)} testing samples from {thread_dir}")"""

# Check if any datasets were found and create DataLoaders
"""if train_datasets:
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_datasets),
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Total training samples: {len(train_loader.dataset)}")
else:
    print("Training datasets:", train_datasets)  # Debugging
    raise RuntimeError("No training data found in specified directories.")

if test_datasets:
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(test_datasets),
        batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Total testing samples: {len(test_loader.dataset)}")
else:
    print("Testing datasets:", test_datasets)  # Debugging
    raise RuntimeError("No testing data found in specified directories.")"""



train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=args.batch_size, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=args.batch_size, shuffle=False, num_workers=2)

model = VAE(3, LSIZE).to(device)
optimizer = optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    BCE = F.mse_loss(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD

def train(epoch):
    """ One training epoch """
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

def test():
    """ One test epoch """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

vae_dir = join(args.logdir, 'vae')
if not exists(vae_dir):
    mkdir(vae_dir)
    mkdir(join(vae_dir, 'samples'))

reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])

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
                       join(vae_dir, 'samples/sample_' + str(epoch) + '_a.png'))

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break
