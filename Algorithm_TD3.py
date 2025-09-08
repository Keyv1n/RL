import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import pickle

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


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, lr_actor,hidden_dim=256):
        super().__init__()

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
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        init_weight(self.fc4, initializer="xavier uniform")
        self.fc4.bias.data.zero_()

        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        lstm_input = observation.unsqueeze(1)
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        x = self.layer_norm1(lstm_output)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.max_action * torch.tanh(self.fc4(x))  # [0,max]

        return x

    def save_checkpoint(self, name: str):
        print('... saving checkpoint ...')
        checkpoint_file = os.path.join(name)
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, name: str):
        print('... loading checkpoint ...')
        checkpoint_file = os.path.join(name)
        self.load_state_dict(torch.load(checkpoint_file))


class CriticNetwork(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, lr_critic,hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim + action_dim, hidden_size=hidden_dim, batch_first=True)
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

        self.fc4 = nn.Linear(hidden_dim, 1)
        init_weight(self.fc4, initializer="xavier uniform")
        self.fc4.bias.data.zero_()

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation, action):
        lstm_input = torch.cat([observation, action], dim=-1).unsqueeze(1)
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        lstm_output = self.layer_norm1(lstm_output)

        x = F.relu(self.fc1(lstm_output))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def save_checkpoint(self, name: str):
        print('... saving checkpoint ...')
        checkpoint_file = os.path.join(name)
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, name: str):
        print('... loading checkpoint ...')
        checkpoint_file = os.path.join(name)
        self.load_state_dict(torch.load(checkpoint_file))


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


class TD3(object):
    def __init__(self, enviro, lr_actor, lr_critic, buffer, warmup=1000, update_actor_step=2,
                 gamma=0.99, tau=0.005, batch_size=64, noise=0.2):
        """
        enviro,
        state_dim, action_dim,
        lr_actor, lr_critic, buffer,
        warmup=1000, update_actor_step=2,
        gamma=0.99, tau=0.005, batch_size=64,
        noise=0.01, device="cuda"
        """
        self.lr = lr_actor
        self.a_loss = 0
        self.c_loss = 0
        self.policy_loss = []
        self.critics_loss = []
        self.policy_losses = []
        self.critics_losses = []
        self.learn_step_cntr = 0
        self.enviro = enviro
        self.state_dim = enviro.observation_space.shape[0]
        self.action_dim = enviro.action_space.shape[0]
        self.buffer_state_dim = enviro.observation_space.shape[0]
        self.actor = ActorNetwork(state_dim=self.state_dim, action_dim=self.action_dim, lr_actor=lr_actor)
        self.critic1 = CriticNetwork(state_dim=self.state_dim, action_dim=self.action_dim, lr_critic=lr_critic)
        self.critic2 = CriticNetwork(state_dim=self.state_dim, action_dim=self.action_dim, lr_critic=lr_critic)
        self.target_actor = ActorNetwork(state_dim=self.state_dim, action_dim=self.action_dim,lr_actor=lr_actor)
        self.target_critic1 = CriticNetwork(state_dim=self.state_dim, action_dim=self.action_dim, lr_critic=lr_critic)
        self.target_critic2 = CriticNetwork(state_dim=self.state_dim, action_dim=self.action_dim, lr_critic=lr_critic)

        # Initialize target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(state_dim=self.buffer_state_dim, action_dim=self.action_dim, buffer_size=buffer)
        self.noise = noise
        self.warmup = warmup
        self.time_step = 0
        self.learn_step_cntr = 0
        self.update_actor_iter = update_actor_step
        self.seed = np.random.seed(seed)

    def remember(self, state, action, reward, new_state, terminal):
        self.replay_buffer.store(state, action, reward, new_state, terminal)

    def choose_action(self, observation,noise=True):
        if self.time_step < self.warmup:
            mu = torch.tensor(np.random.normal(scale=self.noise,size=(self.action_dim,))).to(self.actor.device)
        else:
            observation = observation.to(self.actor.device).clone().detach()
            state = observation.unsqueeze(0)
            with torch.no_grad():
                mu = self.actor(state).to(self.actor.device)
                if noise:
                    mu = mu + torch.tensor(np.random.normal(scale=self.noise , size=mu.shape),dtype=torch.float).to(self.actor.device)
        mu_p = torch.clamp(mu, -1,1) # self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_p.cpu().detach().numpy()

    def process(self):
        if self.replay_buffer.count < 2*self.batch_size:
            return
        obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)

        reward = reward.clone().detach().to(self.critic1.device)
        done = done.clone().detach().to(self.critic1.device)
        next_obs = next_obs.clone().detach().to(self.critic1.device)
        action = action.clone().detach().to(self.critic1.device)
        obs = obs.clone().detach().to(self.critic1.device)
        # next_obs = next_obs.permute(0, 3, 1, 2)
        # obs = obs.permute(0, 3, 1, 2)
        with torch.no_grad():
            next_actions = self.target_actor(next_obs)
            next_actions = next_actions + torch.clamp(torch.tensor(np.random.normal(scale=0.2,size=next_actions.shape),dtype=torch.float32), -0.5, 0.5).to(next_actions.device)#scale std
            next_actions = torch.clamp(next_actions,-1,1) # self.min_action[0], self.max_action[0])
            target_critic_value1 = self.target_critic1(next_obs, next_actions)
            target_critic_value2 = self.target_critic2(next_obs, next_actions)
            target_critic_value = torch.min(target_critic_value1, target_critic_value2)
            target_critic_value = reward +self.gamma*target_critic_value*(1-done)
            target_critic_value = target_critic_value.detach()

        critic_value1 = self.critic1(obs, action)
        critic_value2 = self.critic2(obs, action)

        # --------------------------------
        # critic
        # --------------------------------
        self.critic1.train()
        self.critic2.train()

        critic_loss1 = F.mse_loss(critic_value1, target_critic_value) # mse(prediction, target)
        critic_loss2 = F.mse_loss(critic_value2, target_critic_value)
        ''''
        self.critic1.optimizer.zero_grad()
        critic_loss1.backward()
        self.critic1.optimizer.step()

        self.critic2.optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2.optimizer.step()
        '''
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss = (critic_loss1+critic_loss2)*.5
        self.critics_loss= critic_loss
        self.critics_losses.append(critic_loss.item())
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()
        self.learn_step_cntr += 1
        # --------------------------------
        # actor
        # --------------------------------
        if self.learn_step_cntr % self.update_actor_iter == 0:

            self.actor.train()

            actor_loss = self.critic1(obs, self.actor(obs))
            actor_loss = -torch.mean(actor_loss)
            self.policy_loss = actor_loss
            self.policy_losses.append(actor_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_network_parameters()

    def update_network_parameters(self):

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self,name):
        model_dir = f'model/{name}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.actor.save_checkpoint(os.path.join(model_dir,'_actor.pth'))
        self.target_actor.save_checkpoint(os.path.join(model_dir,'_actor_target.pth'))
        self.critic1.save_checkpoint(os.path.join(model_dir,'_critic1.pth'))
        self.target_critic1.save_checkpoint(os.path.join(model_dir,'_target_critic1.pth'))
        self.critic2.save_checkpoint(os.path.join(model_dir,'_critic2.pth'))
        self.target_critic2.save_checkpoint(os.path.join(model_dir,'_target_critic2.pth'))

    def load_models(self,name):

        self.actor.load_checkpoint(name+'_actor.pth')
        self.target_actor.load_checkpoint(name+'_actor_target.pth')
        self.critic1.load_checkpoint(name+'_critic1.pth')
        self.target_critic1.load_checkpoint(name+'_target_critic1.pth')
        self.critic2.load_checkpoint(name+'_critic2.pth')
        self.target_critic2.load_checkpoint(name+'_target_critic2.pth')

    def save_buffer(self, name):
        model_dir = f'model/{name}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.replay_buffer.save_buffer(os.path.join(model_dir, 'buffer.pkl'))

    def load_buffer(self, name):
        self.replay_buffer = ReplayBuffer.load_buffer(name+'buffer.pkl')

    def learn(self,step,n_step_update,tensorboard: str = './logdir'):
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
            action = self.choose_action(observation,noise=True)
            # print(action,"action")
            next_state, reward, done, _ = self.enviro.step(action)
            # print(reward, "reward")
            self.remember(observation, action, reward, next_state, done)
            if step_counter % n_step_update == 0:
                self.process()
            step_counter += 1
            episode_step += 1
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
                    print('   episode len man :', str(sum(step_per_episode[-1:])))
                    print('   episode rew man :', str(sum(reward_per_episode[-1:])))
                    print('   learning rate   :', str(self.lr))
                    print(30 * "-")
                    print("\n")
                observation = self.enviro.reset()
            else:
                observation = next_state
            writer.add_scalar("reward/episode_reward", (sum(reward_per_episode[-5:])) / 5, global_step=episode)
            writer.add_scalar("reward/episode_reward step", (sum(reward_per_episode[-5:])) / 5,
                              global_step=step_counter)
            writer.add_scalar("LOSS/policy", (sum(self.policy_losses[-5:])) / 5, global_step=step_counter)
            writer.add_scalar("LOSS/critic", (sum(self.critics_losses[-5:])) / 5, global_step=step_counter)
            writer.add_scalars("scalar1", {'learning rate': self.lr}, global_step=step_counter)
            writer.add_scalars("scalar2", {'acc-x': self.enviro.accel[0], 'speed': self.enviro.speed()},
                               global_step=step_counter)
        for name, param in self.actor.named_parameters():
            writer.add_histogram(f'Actor/{name}', param, step_counter)
        for name, param in self.critic1.named_parameters():
            writer.add_histogram(f'Critic1/{name}', param, step_counter)
        for name, param in self.critic2.named_parameters():
            writer.add_histogram(f'Critic2/{name}', param, step_counter)
        writer.close()
        print(30 * "*")
        print(' episode ', step_counter, '\n training avg %.3f' % np.mean(reward_per_episode),
              '\n training max %.3f' % np.max(reward_per_episode))
        print(30 * "*")

    def evaluate(self, tensorboard: str = './logdir',step=5):
        self.load_models("D:/00-paper/model/TD3_seed42_lr0005_rs02_train30000/")
        writer = SummaryWriter(log_dir=tensorboard)
        np.random.seed(0)
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        with torch.no_grad():
            for episode in range(step):
                state = self.enviro.reset()
                done = False
                episode_reward = 0
                step_counter = 0
                while not done:
                    action = self.choose_action(state,noise=False)
                    next_state, reward, done, _ = self.enviro.step(action)
                    episode_reward += reward
                    state = next_state
                    step_counter += 1
                    writer.add_scalars(f"scalar2/{episode}",
                                       {'acc-x': self.enviro.accel[0], 'speed': self.enviro.speed()},
                                       global_step=step_counter)
                print(f"Episode {episode + 1}, Reward: {episode_reward}")