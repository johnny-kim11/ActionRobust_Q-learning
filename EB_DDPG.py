import torch
import torch.nn as nn
import copy
from models import EB_Model

class EB_DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.actor_lr = 1e-4
        self.critic_lr = 5e-4 #5e-4

        self.critic = EB_Model(state_dim, action_dim, max_action)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.MseLoss = nn.MSELoss()

    def choose_action(self, s, mode):
        
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if mode == 'RS':
            a = self.critic.sample_action_RS(s).data.numpy().flatten()
        else:
            a = self.critic.sample_action_LD(s).data.numpy().flatten()
        return a

    def learn(self, relay_buffer, mode):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size) 

        # Compute the target Q
        if mode == 'RS':
            target_sample_action = self.critic_target.sample_action_RS(batch_s_) # Change the sampling method
        else:
            target_sample_action = self.critic_target.sample_action_LD(batch_s_)
        Q_ = self.critic_target.energy(batch_s_, target_sample_action)
        target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_
        target_Q = target_Q.detach()

        # Compute the current Q and the critic loss
        current_Q = self.critic.energy(batch_s, batch_a)
        critic_loss = self.MseLoss(target_Q, current_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
            
    def save(self, checkpoint_path):
        torch.save(self.critic.state_dict(), checkpoint_path)
        
    def load(self, checkpoint_path):
        self.critic.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        