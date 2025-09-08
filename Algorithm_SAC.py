import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


# weight -----------------------------------------------
seed=42
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)

# CriticNetwork--------------------------------------------------------


class CriticNetwork(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, lr_critic,hidden_dim=256):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=state_dim + action_dim, hidden_size=hidden_dim, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        # -----------------Q1-----------------------------
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        init_weight(self.fc1)
        self.fc1.bias.data.zero_()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        init_weight(self.fc2)
        self.fc2.bias.data.zero_()
        self.fc3 = nn.Linear(hidden_dim, 1)
        init_weight(self.fc3, initializer="xavier uniform")
        self.fc3.bias.data.zero_()

        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        init_weight(self.fc4)
        self.fc4.bias.data.zero_()
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        init_weight(self.fc5)
        self.fc5.bias.data.zero_()
        self.fc6 = nn.Linear(hidden_dim, 1)
        init_weight(self.fc6, initializer="xavier uniform")
        self.fc6.bias.data.zero_()

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation, action):

        lstm_input = torch.cat([observation, action],dim=-1).unsqueeze(1)
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        lstm_output= self.layer_norm1(lstm_output)

        x = F.relu(self.fc1(lstm_output))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        y = F.relu(self.fc4(lstm_output))
        y = F.relu(self.fc5(y))
        y = self.fc6(y)
        return x,y

    def save_checkpoint(self, name: str):
        print('... saving checkpoint ...')
        checkpoint_file = os.path.join(name)
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, name: str):
        print('... loading checkpoint ...')
        checkpoint_file = os.path.join(name)
        self.load_state_dict(torch.load(checkpoint_file))


# ActorNetwork--------------------------------------------------------


class ActorNetwork(nn.Module):
    def __init__(self,  state_dim, action_dim, lr_actor,hidden_dim=256):
        super().__init__()
        self.reparam_noise = 20
        self.max_action = 1
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        init_weight(self.fc1)
        self.fc1.bias.data.zero_()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        init_weight(self.fc2)
        self.fc2.bias.data.zero_()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        init_weight(self.fc3)
        self.fc3.bias.data.zero_()
        self.dropout = nn.Dropout(p=0.25)
        self.mu = nn.Linear(hidden_dim, action_dim)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()
        self.sigma = nn.Linear(hidden_dim, action_dim)
        init_weight(self.sigma, initializer="xavier uniform")
        self.sigma.bias.data.zero_()
        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):

        lstm_input = observation.unsqueeze(1)
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        x = self.layer_norm1(lstm_output)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        prob =F.relu(self.fc3(x))
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=2).exp()
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        distribution = Normal(mu, sigma)

        if reparameterize:
            actions = distribution.rsample()
        else:
            actions = distribution.sample()

        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)    #sigmoid = [0,1] action space
        log_probs = distribution.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+ 1e-6)
        log_probs = log_probs.sum(-1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self, name: str):
        print('... saving checkpoint ...')
        checkpoint_file = os.path.join(name)
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, name: str):
        print('... loading checkpoint ...')
        checkpoint_file = os.path.join(name)
        self.load_state_dict(torch.load(checkpoint_file))


# ReplayBuffer--------------------------------------------------------


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, buffer_size=int(1e6)):
        self.max_size = buffer_size
        self.count = 0
        self.size = 0
        self.obs = np.zeros((self.max_size, state_dim))
        self.act = np.zeros((self.max_size, action_dim))
        self.rew = np.zeros((self.max_size, 1))
        self.n_obs = np.zeros((self.max_size, state_dim))
        self.down = np.zeros((self.max_size, 1))
        self.seed = np.random.seed(seed)

    def store(self, obs, act, rew, n_obs, down):
        self.obs[self.count] = obs.cpu().detach().numpy()
        self.act[self.count] = act
        self.rew[self.count] = rew
        self.n_obs[self.count] = n_obs.cpu().detach().numpy()
        self.down[self.count] = down
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  tra 34   Ensitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_obs = torch.tensor(self.obs[index], dtype=torch.float)
        batch_act = torch.tensor(self.act[index], dtype=torch.float)
        batch_rew = torch.tensor(self.rew[index], dtype=torch.float)
        batch_n_obs = torch.tensor(self.n_obs[index], dtype=torch.float)
        batch_down = torch.tensor(self.down[index], dtype=torch.float)
        return batch_obs, batch_act, batch_rew, batch_n_obs, batch_down

    def save_buffer(self,name: str):
        path = os.path.join(name)
        buffer_data = {
            'obs': self.obs,
            'act': self.act,
            'rew': self.rew,
            'n_obs': self.n_obs,
            'down': self.down,
            'count': self.count,
            'size': self.size,
            'max_size': self.max_size}
        with open(path, 'wb') as f:
            pickle.dump(buffer_data, f)
        print('... saving buffer ...')

    @staticmethod
    def load_buffer(name: str):
        path = os.path.join(name)
        with open(path, 'rb') as f:
            buffer_data = pickle.load(f)

        buffer = ReplayBuffer(state_dim=buffer_data['obs'].shape[1], action_dim=buffer_data['act'].shape[1],
                              buffer_size=buffer_data['max_size'])

        buffer.obs = buffer_data['obs']
        buffer.act = buffer_data['act']
        buffer.rew = buffer_data['rew']
        buffer.n_obs = buffer_data['n_obs']
        buffer.down = buffer_data['down']
        buffer.count = buffer_data['count']
        buffer.size = buffer_data['size']
        print('... load buffer ...')

        return buffer


# SAC--------------------------------------------------------


class SAC(object):
    def __init__(self, lr_rate=.001, enviro=None, adaptive_alpha=True,alpha=1,
                 gamma=0.99, buffer=100000, tau=0.005, update_actor_step=1,
                 batch_size=256, reward_scale=2):
        """
        lr_actor=.001, lr_critic=.001,
         enviro=None,
        gamma=0.99,
        buffer=100000, tau=0.005,
        batch_size=256, reward_scale=2
        """
        self.gamma = gamma
        self.tau = tau
        self.learn_step_cntr = 0
        self.update_actor_iter = update_actor_step
        self.lr = lr_rate

        self.policy_loss = []
        self.critics_loss = []
        self.policy_losses = []
        self.critics_losses = []
        self.alpha_losses = []
        self.alpha_loss = []
        self.alpha=[]
        self.alphas = []

        self.batch_size = batch_size
        self.enviro = enviro
        self.state_dim = enviro.observation_space.shape[0]
        self.action_dim = enviro.action_space.shape[0]
        self.buffer_state_dim = enviro.observation_space.shape[0]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')

        self.adaptive_alpha = adaptive_alpha
        if self.adaptive_alpha:
            self.target_entropy = -torch.prod(torch.tensor(enviro.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1,requires_grad=True,device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=.0003)
            self.alpha = (self.log_alpha.exp())*alpha
        else:
            self.alpha=alpha

        # self.lr_scheduler =torch.optim.lr_scheduler.StepLR(self.actor , step_size=10000,gamma=0.9)
        self.replay_buffer = ReplayBuffer(state_dim=self.buffer_state_dim, action_dim=self.action_dim,buffer_size=buffer)
        self.actor = ActorNetwork(lr_actor=lr_rate, state_dim=self.state_dim, action_dim=self.action_dim)
        self.critic = CriticNetwork(lr_critic=lr_rate, state_dim=self.state_dim, action_dim=self.action_dim)
        # self.critic2 = CriticNetwork(lr_critic=lr_rate, state_dim=self.state_dim, action_dim=self.action_dim)
        self.target_critic = CriticNetwork(lr_critic=lr_rate, state_dim=self.state_dim, action_dim=self.action_dim)
        # self.target_critic2 = CriticNetwork(lr_critic=lr_rate, state_dim=self.state_dim, action_dim=self.action_dim)
        self.scale = reward_scale
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.seed = np.random.seed(seed)

    def choose_action(self, observation):

        state = observation.to(self.actor.device).clone().detach()
        state=state.unsqueeze(0)
        actions, _ = self.actor.sample_normal(state, reparameterize=True)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.replay_buffer.store(state, action, reward, new_state, done)

    def update_network_parameters(self):
        tau = self.tau
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        # for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
        #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models(self,name):
        model_dir = f'model/{name}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.actor.save_checkpoint(os.path.join(model_dir, '_actor.pth'))
        self.critic.save_checkpoint(os.path.join(model_dir, '_critic.pth'))
        # self.critic2.save_checkpoint(os.path.join(model_dir, '_critic2.pth'))
        self.target_critic.save_checkpoint(os.path.join(model_dir, '_target_critic.pth'))
        # self.target_critic2.save_checkpoint(os.path.join(model_dir, '_target_critic2.pth'))

    def load_models(self,name):
        self.actor.load_checkpoint(name+ '_actor.pth')
        self.critic.load_checkpoint(name+ '_critic.pth')
        # self.critic2.load_checkpoint(name + '_critic2.pth')
        self.target_critic.load_checkpoint(name+ '_target_critic.pth')
        # self.target_critic2.load_checkpoint(name + '_target_critic2.pth')

    def save_buffer(self, name):
        model_dir = f'model/{name}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.replay_buffer.save_buffer(os.path.join(model_dir, 'buffer.pkl'))

    def load_buffer(self, name):
        self.replay_buffer = ReplayBuffer.load_buffer(name+'buffer.pkl')

    def process(self):
        if self.replay_buffer.count < 2*self.batch_size:
            return
        state, action, reward, new_state, done = self.replay_buffer.sample(self.batch_size)
        reward = reward.clone().detach().to(self.device)
        done = done.clone().detach().to(self.device)
        next_obs = new_state.clone().detach().to(self.device)
        obs = state.clone().detach().to(self.device)
        action = action.clone().detach().to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample_normal(next_obs, reparameterize=True)
            q1_target,q2_target = self.target_critic(next_obs, next_actions)
            critic_value = torch.min(q1_target, q2_target)- self.alpha * next_log_probs
            target_Q = self.scale * reward + self.gamma * critic_value * (1 - done)

        q1,q2 = self.critic(obs, action)
        critic_loss = 0.5*F.mse_loss(q1, target_Q) + 0.5*F.mse_loss(q2, target_Q)
        self.critics_loss=critic_loss
        self.critics_losses.append(critic_loss.item())
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        # -------------------- actor--------------

        self.alphas.append(self.alpha)
        actions, log_prob = self.actor.sample_normal(obs, reparameterize=True)
        q1_policy,q2_policy = self.critic(obs, actions)
        policy_value = torch.min(q1_policy, q2_policy)
        actor_loss = (self.alpha * log_prob - policy_value).mean()
        self.policy_loss=actor_loss
        self.policy_losses.append(actor_loss.item())
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        if self.adaptive_alpha:
            log_prob=log_prob.sum(-1)
            alpha_loss = (self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_loss=alpha_loss
            self.alpha_losses.append(alpha_loss.item())
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().clamp(min=0.001)
            self.alphas.append(self.alpha)

        self.update_network_parameters()

    def learn(self,step,n_step_update,tensorboard: str = './logdir'):

        # self.load_models("D:/00- thesis/code/New Env/model/SAC_1735310334/SAC_50000.0/")
        # self.load_buffer("D:/00- thesis/code/New Env/model/SAC_1735310334/SAC_50000.0/")
        writer = SummaryWriter(log_dir=tensorboard)
        np.random.seed(0)
        step_per_episode = []
        reward_per_episode = [0]
        step_counter = 0
        episode_step = 0
        episode = 0
        score = 0
        observation = self.enviro.reset()
        while step_counter < step:
            action = self.choose_action(observation)
            # print(action, "action")
            next_state, reward, done, _ = self.enviro.step(action)
            # print(reward, "reward")
            self.remember(observation, action, reward, next_state, done)
            if step_counter % n_step_update == 0:
                self.process()
            step_counter +=1
            episode_step +=1
            score += reward
            if done:
                episode += 1
                step_per_episode.append(episode_step)
                reward_per_episode.append(score)
                episode_step = 0
                score = 0
                if episode % 1 == 0:
                    print(30 * "-")
                    print('   num episode     :', str(episode))
                    print('   policy loss     :', str(self.policy_loss))
                    print('   critic loss     :', str(self.critics_loss))
                    print('   alpha           :', str(self.alpha))
                    print('   alpha loss      :', str(self.alpha_loss))
                    print('   episode len man :', str(sum(step_per_episode[-1:])))
                    print('   episode rew man :', str(sum(reward_per_episode[-1:])))
                    print('   learning rate   :', str(self.lr))
                    print(30 * "-")
                    print("\n")
                observation = self.enviro.reset()
            else:
                observation = next_state
            writer.add_scalar("reward/episode_reward", (sum(reward_per_episode[-5:]))/5, global_step=episode)
            writer.add_scalar("reward/episode_reward step", (sum(reward_per_episode[-5:])) / 5, global_step=step_counter)
            writer.add_scalar("LOSS/coe alpha", (sum(self.alphas[-5:]))/5, global_step=step_counter)
            writer.add_scalar("LOSS/alpha", (sum(self.alpha_losses[-5:]))/5, global_step=step_counter)
            writer.add_scalar("LOSS/policy",(sum(self.policy_losses[-5:]))/5, global_step=step_counter)
            writer.add_scalar("LOSS/critic", (sum(self.critics_losses[-5:]))/5, global_step=step_counter)
            writer.add_scalars("scalar1", {'learning rate': self.lr}, global_step=step_counter)
            writer.add_scalars("scalar2", {'acc-x': self.enviro.accel[0], 'speed': self.enviro.speed()},global_step=step_counter)
        for name, param in self.actor.named_parameters():
            writer.add_histogram(f'Actor/{name}', param, step_counter)
        for name, param in self.critic.named_parameters():
            writer.add_histogram(f'Critic/{name}', param, step_counter)
        writer.close()
        print(30 * "*")
        print(' episode ', step_counter, '\n training avg %.3f' % np.mean(reward_per_episode),
              '\n training max %.3f' % np.max(reward_per_episode))
        print(30 * "*")

    def evaluate(self, tensorboard: str = './logdir',step=5):
        self.load_models("D:/00-paper/model/SAC_seed42_lr0005_rs02_train30000/")
        # self.load_models("D:/00-paper/model/SAC_seed42_lr0005_rs02_train30000/")

        print("YES")

        writer = SummaryWriter(log_dir=tensorboard)
        np.random.seed(0)
        self.actor.eval()
        with torch.no_grad():
            for episode in range(step):
                state = self.enviro.reset()
                done = False
                episode_reward = 0
                step_counter = 0
                while not done:
                    action = self.choose_action(state)
                    next_state, reward, done, _ = self.enviro.step(action)
                    episode_reward += reward
                    state = next_state
                    step_counter += 1
                    writer.add_scalars(f"scalar2/{episode}", {'acc-x': self.enviro.accel[0], 'speed': self.enviro.speed()},
                                   global_step=step_counter)
                print(f"Episode {episode + 1}, Reward: {episode_reward}")


