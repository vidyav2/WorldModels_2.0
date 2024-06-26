""" Various auxiliary utilities """
import math
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
from models import MDRNNCell, VAE, Controller
import gym
import gym.envs.box2d

import cv2


# A bit dirty: manually change size of car racing env
gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 96, 96

# Hardcoded for now
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 96

# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions


def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    def __init__(self, mdir, device, time_limit):
        vae_file, rnn_file, ctrl_file = [join(mdir, m, 'best.tar') for m in ['vaeNew', 'mdrnn', 'ctrl']]
        assert exists(vae_file) and exists(rnn_file), "Either vae or mdrnn is untrained."
        
        vae_state, rnn_state = [torch.load(fname, map_location={'cuda:0': str(device)}) for fname in (vae_file, rnn_file)]
        
        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} with test loss {}".format(m, s['epoch'], s['precision']))
        
        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])
        print("VAE has {} parameters".format(sum([p.numel() for p in self.vae.parameters()])))
        
        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict({k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
        print("MDRNN has {} parameters".format(sum([p.numel() for p in self.mdrnn.parameters()])))
        
        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)
        print("Controller has {} parameters".format(sum([p.numel() for p in self.controller.parameters()])))
        
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])
        
        self.env = gym.make('CarRacing-v2')
        self.device = device
        self.time_limit = time_limit
        self.writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 50, (64, 64))

    def get_action_and_transition(self, obs, hidden):
        recon_obs, latent_mu, _ = self.vae(obs)
        recon_obs = recon_obs.squeeze().permute(1, 2, 0).cpu().numpy()
        recon_obs = (recon_obs * 255).astype(np.uint8)
        recon_obs = cv2.cvtColor(recon_obs, cv2.COLOR_RGB2BGR)
        self.writer.write(recon_obs)
        action = self.controller(latent_mu, hidden[0])
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, params, render=False):
        if params is not None:
            load_parameters(params, self.controller)
        obs, _ = self.env.reset()
        hidden = [torch.zeros(1, RSIZE).to(self.device) for _ in range(2)]
        cumulative = 0
        i = 0
        while True:
            print(f"Rendering frame {i}", end='\r')
            self.writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            obs = transform(obs[:84, :, :]).unsqueeze(0).to(self.device)
            action, hidden = self.get_action_and_transition(obs, hidden)
            obs, reward, done, _, _ = self.env.step(action)
            if render:
                self.env.render()
            cumulative += reward
            if done or i > self.time_limit:
                self.writer.release()
                return -cumulative
            i += 1

