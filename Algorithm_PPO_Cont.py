import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.distributions.normal import Normal
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim


# ActorNetwork--------------------------------------------------------


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, lr_actor, hidden_dim=128):
        super().__init__()
        self.max_action = 1
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=2)

        self.flatten_size = 32 * 15 * 15
        self.fc1 = nn.Linear(self.flatten_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.layer_norm1 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        x = F.relu(self.conv1(observation))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))

        lstm_input = x.unsqueeze(1)
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]

        # x = torch.relu(self.fc1(lstm_output))
        # optional
        # x = self.dropout(x)
        # x = self.layer_norm1(x)
        x = torch.relu(self.fc2(lstm_output))

        # discrete
        # -------------
        # dist = nn.Softmax(dim=-1)(self.fc3(x))
        # dist = Categorical(dist)
        # --------------
        # continues
        dist = self.fc3(x)
        std = torch.exp(self.log_std)
        dist = Normal(dist, std)
        return dist

    def save_checkpoint(self, name: str):
        print('... saving checkpoint ...')
        checkpoint_file = os.path.join(name)
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, name: str):
        print('... loading checkpoint ...')
        checkpoint_file = os.path.join(name)
        self.load_state_dict(torch.load(checkpoint_file))


# CriticNetwork--------------------------------------------------------


class CriticNetwork(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, lr_critic, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        self.flatten_size = 32 * 15 * 15
        self.fc1 = nn.Linear(self.flatten_size, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        x = F.relu(self.conv1(observation))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
    def __init__(self, state_dim, action_dim):

        self.count = 0
        self.size = 0
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def store(self, observation, action, log_prob, value, reward, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        # self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        # self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def generate_batches(self, batch_size):

        observations = torch.tensor(self.observations, dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.float32)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        indices = np.arange(len(rewards))
        np.random.shuffle(indices)
        batches = [indices[i:i + batch_size] for i in range(0, len(rewards), batch_size)]

        return observations, actions, log_probs, values, rewards, dones, batches

    def clear(self):
        self.observations = []
        self.log_probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []


# PPO--------------------------------------------------------


class PPO(object):
    def __init__(self, enviro, gamma=0.99, lr_actor=0.001, lr_critic=0.001, gae_lambda=0.95,
                 policy_clip=0.1, batch_size=64, n_epochs=10):

        # enviro, action_dim, state_dim,
        # gamma=0.99, lr_actor=0.001,
        # lr_critic=0.001, gae_lambda=0.95,
        # policy_clip=0.1, batch_size=64,
        # n_epochs=10, device="cuda"

        self.lr = lr_actor
        self.enviro = enviro
        self.state_dim = 32
        self.action_dim = enviro.action_space.shape[0]
        self.buffer_state_dim = 32
        self.actor = ActorNetwork(self.state_dim, self.action_dim, lr_actor=lr_actor)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, lr_critic=lr_critic)

        # Initialize target networks

        self.transform = transforms.Compose([transforms.ToTensor(), ])
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.a_loss = 0
        self.c_loss = 0

        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(self.buffer_state_dim,self.action_dim)

    def remember(self, state, action, reward, probs, vals, terminal):
        self.replay_buffer.store(observation=state, action=action, log_prob=probs, value=vals, reward=reward, done=terminal)

    def choose_action(self, observation):

        # observation=self.transform(observation)
        state = observation.to(self.actor.device).clone().detach().unsqueeze(0)

        value = self.critic(state)
        dist = self.actor(state)
        action = dist.sample()

        probs = dist.log_prob(action).sum(dim=-1).item()
        action = action.cpu().numpy()
        value = value.item()

        return action, probs, value

    def process(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.replay_buffer.generate_batches(self.batch_size)
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)

                prob_ratio = new_probs.exp() / old_probs.exp()

                # prob_ratio = (new_probs - old_probs).exp()
                ratio = advantage[batch] * prob_ratio
                clipped_ratio = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]

                actor_loss = -torch.min(ratio, clipped_ratio).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = F.mse_loss(returns, critic_value)

                total_loss = actor_loss + 0.5 * critic_loss
                total_loss = torch.tensor(total_loss, dtype=torch.float, requires_grad=True).to(self.actor.device)

                self.a_loss = total_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.replay_buffer.clear()

    def save_models(self, name):
        model_dir = f'PPO_con/{name}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.actor.save_checkpoint(os.path.join(model_dir, '_actor.pth'))
        self.critic.save_checkpoint(os.path.join(model_dir, '_critic.pth'))

    def load_models(self, name):

        self.actor.load_checkpoint(name+'_actor')
        self.critic.load_checkpoint(name+'_critic')

    def learn(self, step, n_step_update, tensorboard: str = './logdir'):
        writer = SummaryWriter(log_dir=tensorboard)
        np.random.seed(0)
        score_history = []
        for i in range(step):
            observation = self.enviro.reset()
            done = False
            n_steps = 0
            score = 0
            counter = 0
            while not done:
                counter += 1
                action, prob, val = self.choose_action(observation)
                next_state, reward, done, _ = self.enviro.step(action)
                n_steps += 1
                score += reward
                self.remember(state=observation, action=action, probs=prob, vals=val, reward=reward, terminal=done)
                if n_steps % n_step_update == 0:
                    self.process()
                observation = next_state

                writer.add_scalar("reward/per step", score, global_step=counter)
                writer.add_scalar("loss actor/per step", self.a_loss, global_step=counter)
                # writer.add_scalar("loss critic/per step", self.c_loss, global_step=counter)
                writer.add_scalars("control", {'x': self.enviro.accel[0], 'A': self.enviro.avg_kmh,
                                               'd': self.enviro.radar_data[0], 'relate v': self.enviro.radar_data[0]},
                                   global_step=step)

            score_history.append(score)
            print(30 * "-")
            print('   episode         :', str(i))
            print('   max step        :', str(self.enviro.step_counter))
            print('   total_loss      :', str(self.a_loss))

            print('   learning rate   :', str(self.lr))
            print(30 * "-")
            print("\n")
            writer.add_scalar("loss/Total_Loss", self.a_loss, global_step=step)
            writer.add_scalar("reward/Total_reward", score, global_step=step)
        for name, param in self.actor.named_parameters():
            writer.add_histogram(f'Actor/{name}', param, step)
        for name, param in self.critic.named_parameters():
            writer.add_histogram(f'Critic/{name}', param, step)
        print(30 * "*")
        print(' episode ', step, '\n training avg score %.3f' % np.mean(score_history),
              '\n training max score %.3f' % np.max(score_history),
              '\n training min score %.3f' % np.min(score_history))
        print(30 * "*")
