""" Memory model training """
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from utils.misc import save_checkpoint
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
from utils.learning import EarlyStopping, ReduceLROnPlateau
from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss

parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument('--logdir', type=str, help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true', help="Do not reload if specified.")
parser.add_argument('--include_reward', action='store_true', help="Add a reward modelisation term to the loss.")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
BSIZE = 36
SEQ_LEN = 32
epochs = 100
LSIZE = 32

# Loading VAE
vae_file = join(args.logdir, 'vaeNew', 'best.tar')
assert exists(vae_file), "No trained VAE in the logdir..."
state = torch.load(vae_file)
print(f"Loading VAE at epoch {state['epoch']} with test error {state['precision']}")

vae = VAE(3, LSIZE).to(device)
vae.load_state_dict(state['state_dict'])

# Loading MDRNN
rnn_dir = join(args.logdir, 'mdrnn')
rnn_file = join(rnn_dir, 'best.tar')
if not exists(rnn_dir):
    mkdir(rnn_dir)

mdrnn = MDRNN(LSIZE, ASIZE, RSIZE, 5).to(device)
optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

if exists(rnn_file) and not args.noreload:
    rnn_state = torch.load(rnn_file)
    print(f"Loading MDRNN at epoch {rnn_state['epoch']} with test error {rnn_state['precision']}")
    mdrnn.load_state_dict(rnn_state['state_dict'])
    optimizer.load_state_dict(rnn_state['optimizer'])
    scheduler.load_state_dict(rnn_state['scheduler'])
    earlystopping.load_state_dict(rnn_state['earlystopping'])

# Data Loading
transform = transforms.Lambda(lambda x: np.transpose(x, (0, 3, 1, 2)) / 255.0)
train_loader = DataLoader(RolloutSequenceDataset(['datasets/carracing'], SEQ_LEN, transform, buffer_size=30),
                          batch_size=BSIZE, num_workers=8, shuffle=True, drop_last=True)
test_loader = DataLoader(RolloutSequenceDataset(['datasets/carracing'], SEQ_LEN, transform, train=False, buffer_size=10),
                         batch_size=BSIZE, num_workers=8, drop_last=True)

def to_latent(obs, next_obs):
    with torch.no_grad():
        obs, next_obs = [F.interpolate(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE, mode='bilinear', align_corners=True)
                         for x in (obs, next_obs)]
        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [vae(x)[1:] for x in (obs, next_obs)]

        total_elements = obs_mu.nelement()  # Total elements in the tensor
        calculated_bsize = total_elements // (SEQ_LEN * LSIZE)  # Calculate BSIZE dynamically

        try:
            latent_obs, latent_next_obs = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(calculated_bsize, SEQ_LEN, LSIZE)
                for x_mu, x_logsigma in [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]
            ]
        except RuntimeError as e:
            print(f"Reshape error with calculated BSIZE: {calculated_bsize}")
            raise e
        
    return latent_obs, latent_next_obs

def get_loss(latent_obs, action, reward, terminal, latent_next_obs, include_reward: bool):
    if action.size(0) == 0:
        raise ValueError("Action tensor is empty. Check data loading and batching.")

    latent_obs, action, reward, terminal, latent_next_obs = [
        arr.transpose(1, 0) for arr in [latent_obs, action, reward, terminal, latent_next_obs]
    ]
    action = action.repeat(1, latent_obs.size(1) // action.size(1), 1)

    # Print shapes for debugging
    print(f"latent_obs shape before mdrnn: {latent_obs.shape}, action shape before mdrnn: {action.shape}")

    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = F.binary_cross_entropy_with_logits(ds, terminal)
    mse = F.mse_loss(rs, reward) if include_reward else 0
    loss = (gmm + bce + mse) / (LSIZE + 2 if include_reward else LSIZE + 1)
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


def data_pass(epoch, train, include_reward):  # pylint: disable=too-many-locals
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader

    loader.dataset.load_next_buffer()
    cum_loss = cum_gmm = cum_bce = cum_mse = 0

    pbar = tqdm(total=len(loader.dataset), desc=f"Epoch {epoch}")
    for i, data in enumerate(loader):
        obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]
        latent_obs, latent_next_obs = to_latent(obs, next_obs)

        # Check and handle empty action tensor
        if action.size(0) == 0:
            raise ValueError("Action tensor is empty. Check data loading and batching.")

        # Ensure consistent batch sizes
        if latent_obs.size(0) != action.size(0):
            print(f"Batch size mismatch: latent_obs size = {latent_obs.size(0)}, action size = {action.size(0)}")
            raise ValueError("Mismatch in sizes between latent observations and actions.")

        action = action.transpose(1, 0)
        if latent_obs.size(1) % action.size(1) != 0:
            print(f"Action size mismatch: latent_obs size = {latent_obs.size(1)}, action size = {action.size(1)}")
            raise ValueError("Mismatch in sizes between latent observations and actions.")

        action = action.repeat(1, latent_obs.size(1) // action.size(1), 1)

        # Print shapes for debugging
        print(f"latent_obs shape: {latent_obs.shape}, action shape: {action.shape}")

        if train:
            losses = get_loss(latent_obs, action, reward, terminal, latent_next_obs, include_reward)
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(latent_obs, action, reward, terminal, latent_next_obs, include_reward)

        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()
        cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else losses['mse']

        pbar.set_postfix_str(f"loss={cum_loss / (i + 1):.6f} bce={cum_bce / (i + 1):.6f} "
                             f"gmm={cum_gmm / LSIZE / (i + 1):.6f} mse={cum_mse / (i + 1):.6f}")
        pbar.update(BSIZE)
    pbar.close()
    return cum_loss * BSIZE / len(loader.dataset)


train = partial(data_pass, train=True, include_reward=args.include_reward)
test = partial(data_pass, train=False, include_reward=args.include_reward)

cur_best = None
for e in range(epochs):
    train(e)
    test_loss = test(e)
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
    checkpoint_fname = join(rnn_dir, 'checkpoint.tar')
    save_checkpoint({
        "state_dict": mdrnn.state_dict(),
        "optimizer": optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict(),
        "precision": test_loss,
        "epoch": e}, is_best, checkpoint_fname, rnn_file)

    if earlystopping.stop:
        print(f"End of Training because of early stopping at epoch {e}")
        break
