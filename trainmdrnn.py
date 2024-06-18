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
BSIZE = 16
SEQ_LEN = 32
epochs = 30

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
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (BSIZE, SEQ_LEN, 3, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, 3, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        obs, next_obs = [
            F.interpolate(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE, mode='bilinear', align_corners=True)
            for x in (obs, next_obs)]
        
        print(f"After interpolation - obs: {obs.shape}, next_obs: {next_obs.shape}")
        
        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae(x)[1:] for x in (obs, next_obs)]
        
        print(f"After VAE encoding - obs_mu: {obs_mu.shape}, obs_logsigma: {obs_logsigma.shape}, next_obs_mu: {next_obs_mu.shape}, next_obs_logsigma: {next_obs_logsigma.shape}")
        
        latent_obs, latent_next_obs = [
            (x_mu + x_logsigma.exp() * torch.randn_like(x_mu))
            for x_mu, x_logsigma in
            [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]

        # Print the shape of the tensor before reshaping
        print(f"Before reshaping - latent_obs: {latent_obs.shape}, latent_next_obs: {latent_next_obs.shape}")
        
        # Reshape the tensors to match (BSIZE, SEQ_LEN, LSIZE)
        latent_obs = latent_obs.view(BSIZE, SEQ_LEN, LSIZE)
        latent_next_obs = latent_next_obs.view(BSIZE, SEQ_LEN, LSIZE)
        
        print(f"After reshaping - latent_obs: {latent_obs.shape}, latent_next_obs: {latent_next_obs.shape}")

    return latent_obs, latent_next_obs

def get_loss(latent_obs, action, reward, terminal, latent_next_obs, include_reward: bool):
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args terminal: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    latent_obs, action, reward, terminal, latent_next_obs = [
        arr.transpose(1, 0) for arr in [latent_obs, action, reward, terminal, latent_next_obs]
    ]

    # Ensure actions and latents have the correct dimensions
    latent_obs = latent_obs.reshape(BSIZE, SEQ_LEN, LSIZE)
    latent_next_obs = latent_next_obs.reshape(BSIZE, SEQ_LEN, LSIZE)

    # Concatenate action and latent_obs along the last dimension
    inputs = torch.cat([action, latent_obs], dim=-1)
    
    mus, sigmas, logpi, rs, ds = mdrnn(inputs)
    
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = F.binary_cross_entropy_with_logits(ds, terminal)
    mse = F.mse_loss(rs, reward) if include_reward else 0
    loss = (gmm + bce + mse) / (LSIZE + 2 if include_reward else LSIZE + 1)
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)

def data_pass(epoch, train, include_reward):
    """ One pass through the data """
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader

    loader.dataset.load_next_buffer()

    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0

    pbar = tqdm(total=len(loader.dataset), desc=f"Epoch {epoch}")
    for i, data in enumerate(loader):
        obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]

        # Debugging print statements for loaded data shapes
        print(f"Loaded data shapes - obs: {obs.shape}, action: {action.shape}, reward: {reward.shape}, terminal: {terminal.shape}, next_obs: {next_obs.shape}")

        # Transform obs
        latent_obs, latent_next_obs = to_latent(obs, next_obs)

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
        cum_mse += losses['mse'].item() if include_reward else 0

        if i % 100 == 0 and i > 0:
            print(f"{'Train' if train else 'Test'} Epoch {epoch} | "
                  f"Loss {cum_loss / (i + 1):.6f} "
                  f"gmm={cum_gmm / (i + 1):.6f} "
                  f"bce={cum_bce / (i + 1):.6f} "
                  f"mse={cum_mse / (i + 1):.6f} ({len(loader.dataset)})")
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
