import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, s, a):
        q = F.relu(self.l1(torch.cat([s, a], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q
    
class EB_Model(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(EB_Model, self).__init__()

        self.energy_model = nn.Sequential(
                        nn.Linear(state_dim + action_dim, 256),
                        nn.Tanh(),
                        nn.Linear(256, 256),
                        nn.Tanh(),
                        nn.Linear(256, 1)
                    )
        self.max_action = max_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        
    def forward(self):
        raise NotImplementedError
    
    def energy(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.energy_model(state_action)

    def sample_action_LD(self, state):
        initial_action = np.random.uniform(-self.max_action, self.max_action, state.shape[0] * self.action_dim)
        initial_action = torch.from_numpy(initial_action).float().reshape(state.shape[0], self.action_dim)

        # Langevin Dynamics Setting
        step_size = torch.tensor(0.01)  # Langevin Dynamics의 step size 설정
        num_steps = 200  # 몇 번의 스텝을 실행할 것인지 설정
        initial_action.requires_grad = True

        # Langevin Dynamics Sampling 수행
        for _ in range(num_steps):
            energy = self.energy(state, initial_action)
            x = torch.ones(state.shape[0], 1)
            initial_action.retain_grad()
            energy.backward(torch.ones_like(x), retain_graph=True)
            gradient = initial_action.grad

            noise = torch.randn_like(initial_action) * torch.sqrt(2 * step_size)
            initial_action = initial_action + step_size * gradient + noise/10
            
            initial_action = torch.clip(initial_action, -self.max_action, self.max_action)
        return initial_action
    
    def sample_action_RS(self, state):
        test_action_np = np.random.uniform(-self.max_action, self.max_action, 2000*state.shape[0] * self.action_dim)
        test_action_torch = torch.from_numpy(test_action_np).float().reshape(2000*state.shape[0], self.action_dim)
        repeated_state = torch.repeat_interleave(state, 2000, dim=0)
        q_vals = self.energy(repeated_state, test_action_torch).reshape(state.shape[0], 2000)

        count_accepted_np = np.zeros(state.shape[0]).reshape(-1, 1)
        record_accepted_actions = np.zeros((state.shape[0], self.action_dim))

        with torch.no_grad():
            min_q_vals, _ = torch.min(q_vals, dim=1)
            max_q_vals, _ = torch.max(q_vals, dim=1)
        min_q_vals = min_q_vals.numpy().reshape(-1, 1)
        max_q_vals = max_q_vals.numpy().reshape(-1, 1)
        
        # Max 크기
        M = max_q_vals - min_q_vals

        while True:
            # 임의값 샘플링
            u = np.random.uniform(-self.max_action, self.max_action, self.action_dim*state.shape[0])
            u = torch.from_numpy(u).reshape(state.shape[0], self.action_dim).float()

            # Rejection 계산
            acceptance_rate = (self.energy(state, u).detach().numpy().reshape(state.shape[0], -1) - min_q_vals) / M
            certain_val = np.full((state.shape[0], 1), 0.8) # 0.0 -> 0.9
            random_val = np.random.uniform(0, 1, state.shape[0]).reshape(-1, 1)
            compare_val = np.concatenate((certain_val, random_val), axis=1)
            compare_val = np.max(compare_val, axis=1).reshape(-1, 1)

            accepted_idx, _ = np.where(acceptance_rate > compare_val)
            count_accepted_np[accepted_idx] = True
            record_accepted_actions[accepted_idx] = u[accepted_idx].detach().numpy()
            
            if (count_accepted_np == True).all():
                u = torch.from_numpy(record_accepted_actions).reshape(state.shape[0], self.action_dim).float()
                break
        return u
    
    def calc_log_prob(self, state, action):
        energy = self.energy(state, action)
        x = torch.ones(state.shape[0], 1)
        action.retain_grad()
        energy.backward(torch.ones_like(x))
        gradient = action.grad

        return gradient.mean(dim=1).reshape(-1, 1)