import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
from collections import deque

import random
import enum

df = pd.read_csv('SS00001.csv')
df = df.dropna()
df = df.set_index("Date")

# data preprocessing

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

df['dClose'] = sigmoid(df['Close'] - df['Close'].shift(1))
df['MACD'] = sigmoid(df['MACD'])
df['CCI'] = sigmoid(df['CCI'])
df['RSI'] = np.log(df['RSI'])
df['ADX'] = np.log(df['ADX'])
indicator = df.loc[:,['RSI', 'MACD', 'CCI', 'ADX']]
close = df['Close']
dclose = df['dClose']
data = pd.concat([close, dclose, indicator], axis=1)

train_data = data.loc[(data.index > '2018-01-01') & (data.index <= '2022-12-31'), :]
test_data = data.loc[data.index >= '2023-01-01', :]

buffer_length = 2000
#Hyperparameters
learning_rate = 1.0e-4
gamma         = 0.98
batch_size    = 64
tau = 0.001
noise_scale = 1.5
final_noise_scale = 0.5

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_space, mu = 0, sigma=0.4, theta=.01, scale=0.1):
        self.theta = theta
        self.mu = mu*np.ones(action_space)
        self.sigma = sigma
        self.scale = scale
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) + \
            self.sigma * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x * self.scale

    def reset(self):
        self.x_prev = np.zeros_like(self.mu)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)



class Actions(enum.Enum):
    Hold = 0
    Buy = 1
    Sell = 2

class Trade():
    def __init__(self, data, starting_balance=1000000, episodic_length=20, mode='train'):  
        super(Trade, self).__init__()
        self.data = data
        self.price_column = 'Close'
        self.dClose_column = 'dClose'
        self.indicator_columns = ['RSI', 'MACD', 'CCI', 'ADX']
        self.episodic_length = episodic_length
        self.starting_balance = starting_balance
        self.commission_rate = 0.001
        self.cash = self.starting_balance
        self.shares = 0
        self.total_episodes = len(data) - episodic_length
        self.cur_step = self.next_episode
        self.mode = mode

    def reset(self):
        self.cash = self.starting_balance
        self.shares = 0
        self.cur_step = 0
        return self.next_observation()

    
    def step(self, action):   # assume every day 
        balance = self.cur_balance
        self.cur_step += 1
        self.take_action(action)
        state = self.next_observation()
        reward = self.cur_balance - balance
        done = self.cur_step == self.total_steps - 1
        return state, reward, done
    
    def take_action(self, actions): 
        action = actions.argmax() 
        if action == Actions.Buy.value:
            price = self.cur_close_price * (1 + self.commission_rate)
            if self.cash > price:
                self.cash -= price
                self.shares += 1

        elif action == Actions.Sell.value:
            if self.shares > 0:
                price = self.cur_close_price * (1 - self.commission_rate)
                self.cash += price
                self.shares -= 1

        
    def next_observation(self):
        obs = []
        obs = np.append(obs, [self.cur_dclose_price])
        obs = np.append(obs, [np.log(self.cur_balance), np.log(self.shares+1.0e-6), np.log(self.cur_close_price)])
        return np.append(obs, [self.cur_indicators])
    
    @property
    def next_episode(self):
        return random.randrange(0, self.total_episodes)

    @property
    def cur_indicators(self):
        indicators = self.data[self.indicator_columns]
        return indicators.values[self.cur_step]

    @property
    def cur_dclose_price(self):
        dclose = self.data[self.dClose_column].values
        d = self.cur_step - self.episodic_length
        if d >= 0:
            return dclose[d:self.cur_step]
        else:
            return np.append(np.array(-d*[dclose[0]]), dclose[0:self.cur_step])

    @property
    def total_steps(self):    
        return len(self.data)
    
    @property
    def cur_close_price(self):
        return self.data[self.price_column].iloc[self.cur_step]

    @property
    def cur_balance(self):
        return self.cash + (self.shares * self.cur_close_price)


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)  
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)  
        self.ln2 = nn.LayerNorm(256)
        self.value = nn.Linear(256, action_size) 
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x, softmax_dim = -1):
        x = self.fc1(x)
        x = F.relu(self.ln1(x))
        x = self.fc2(x)
        x = F.relu(self.ln2(x))
        prob = F.softmax(self.value(x), dim=softmax_dim)
        return prob

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)  
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512 + action_size, 256)  
        self.ln2 = nn.LayerNorm(256)
        self.value = nn.Linear(256, 1) 
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, actions):
        x = self.fc1(x)
        x = F.relu(self.ln1(x))
        x = torch.cat((x, actions), 1)
        x = self.fc2(x)
        x = F.relu(self.ln2(x))
        value = F.relu(self.value(x))
        return value
    
class Memory():
    def __init__(self, buffer_len = 1000):
        self.data = deque(maxlen = buffer_len)

    def put_data(self, transition):
        self.data.append(transition)
    
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        mini_batch = random.sample(self.data, batch_size)
        for transition in mini_batch:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
  
def train_net(memory, q, q_target, pi, pi_target):
    loss_pi = []
    loss_q = []

    for i in range(10):
        s, a, r, s_prime, done = memory.make_batch()

        q_batch = q(s, a)
        a_prime_batch = pi_target(s_prime)
        q_prime_batch = r + gamma * done * q_target(s_prime, a_prime_batch)

        value_loss = F.mse_loss(q_batch, q_prime_batch)

        q.optimizer.zero_grad()
        value_loss.backward()
        q.optimizer.step()
        
        policy_loss = -q(s, pi(s))
        policy_loss = policy_loss.mean()
        pi.optimizer.zero_grad()
        policy_loss.backward()
        pi.optimizer.step()

        soft_update(pi_target, pi, tau)
        soft_update(q_target, q, tau)

        loss_pi.append(policy_loss.detach().item())
        loss_q.append(value_loss.detach().item())
        
    return np.mean(loss_pi), np.mean(loss_q)


# select action with para noise on the last layer
def select_action(state, pi, pi_noisy, noise):  
    
    hard_update(pi_noisy, pi)
    actor_params = pi_noisy.state_dict()
    
    param = actor_params['value.bias']
    param += torch.tensor(noise).float()
        
    action = pi_noisy(state)
    return action.detach().numpy()


def train(starting_balance = 100000, window_size = 20, resume_epoch = 0, max_epoch = 2000): 
    mode = 'train'
    env = Trade(train_data, starting_balance, window_size, mode)
    state_size = window_size+7
    action_size = 3
    pi = Actor(state_size, action_size)
    pi_target = Actor(state_size, action_size)
    pi_noisy = Actor(state_size, action_size)
    q = Critic(state_size, action_size)
    q_target = Critic(state_size, action_size)
    memory = Memory(buffer_length)
    noise = OrnsteinUhlenbeckActionNoise(action_size)
    val_loss_history = []
    policy_loss_history = []
    pv_history = []    # portfolio history
    # to continue from the previous saving

    start_epoch = resume_epoch

    if start_epoch > 0:
        pi.load_state_dict(torch.load("DDPGpi_ep" + str(start_epoch)))
        q.load_state_dict(torch.load("DDPGq_ep" + str(start_epoch)))

    hard_update(q_target, q)
    hard_update(pi_target, pi)
    
    pbar = tqdm(range(start_epoch, max_epoch))
    for n_epi in pbar:
        s = env.reset()
        done = False
    
        # dwindling noise 
        noise.scale = (noise_scale - final_noise_scale) * max(0, 3000-n_epi)/3000 + final_noise_scale
        
        # complete episode from start to end to build ReplayBuffer

        # while not done:
        for t in range(1, 100):
            state = torch.from_numpy(s).float()
            prob = select_action(state, pi, pi_noisy, noise())
            s_prime, r, done = env.step(prob)
            memory.put_data((s, prob, r, s_prime, done))

            if len(memory.data)>=128:
                for _ in range(4):
                    policy_loss, val_loss = train_net(memory, q, q_target, pi, pi_target)
                    val_loss_history.append(val_loss)
                    policy_loss_history.append(policy_loss)

            s = s_prime
                
            if done:
                break                     
                

        pv_history.append(env.cur_balance)

        pbar.set_description("%.4f" % env.cur_balance)
        
        if n_epi % 100 == 0:
            torch.save(q.state_dict(), "DDPGq_ep" + str(n_epi))    
            torch.save(pi.state_dict(), "DDPGpi_ep" + str(n_epi))    


    torch.save(q.state_dict(), "DDPGq_epfinal") 
    torch.save(pi.state_dict(), "DDPGpi_epfinal") 

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, sharex = True)

    axs[0].plot(val_loss_history)
    axs[0].set_ylabel('val_loss', fontsize=12)
    axs[0].set_xlabel("update",fontsize=12)

    axs[1].plot(policy_loss_history)
    axs[1].set_ylabel('policy_loss', fontsize=12)
    axs[1].set_xlabel("update",fontsize=12)

    axs[2].plot(pv_history)
    axs[2].set_ylabel('profit', fontsize=12)
    axs[2].set_xlabel("date",fontsize=12)

    plt.savefig('DDPG_S00001_test.png')
    plt.pause(3)
    plt.show()

    
def test(starting_balance = 100000, window_size = 20, model_epi = 'final'):  
    mode = 'test'
    env = Trade(test_data, starting_balance, window_size, mode)
    state_size = window_size + 7
    action_size = 3
    pi = Actor(state_size, action_size)
    q = Critic(state_size, action_size)
    pi.load_state_dict(torch.load("DDPGpi_ep" + str(model_epi)))
    q.load_state_dict(torch.load("DDPGq_ep" + str(model_epi)))
    
    pi.eval()
    q.eval()

    action_history = []
    pv_history = []    # portfolio history

    s = env.reset()
    done = False
    
    # start from any random position of the training data
    while not done:
        state = torch.from_numpy(s).float()
        prob = pi(state)
        s_prime, r, done = env.step(prob)
        s = s_prime
        action_history.append( prob.argmax().item() )
        pv = np.exp(s_prime[window_size])    
        pv_history.append(pv)                   

    print("portfolio: {0}".format(pv))   

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex = True)

    np_actions = np.array(action_history)
    test_close = test_data["Close"].values
    axs[0].plot(test_close)
    index_0 = np.where(np_actions == 0)[0]
    index_1 = np.where(np_actions == 1)[0]
    index_2 = np.where(np_actions == 2)[0]
    
    axs[0].scatter(index_0, test_close[index_0], c='red', label='hold', marker='^')
    axs[0].scatter(index_1, test_close[index_1], c='green', label='buy', marker='>')
    axs[0].scatter(index_2, test_close[index_2], c='blue', label='sell', marker='v')
    axs[0].legend()
    axs[0].set_ylabel('Close', fontsize=22)

    axs[1].plot(pv_history, c='red')
    axs[1].plot(test_data['Close'].values * starting_balance / test_data['Close'].iloc[0], c='black')
    axs[1].set_ylabel('Portfolio', fontsize=22)
    axs[1].set_xlabel("date",fontsize=22)
    plt.savefig('DDPG_S00001_test.png')
    plt.show()
    plt.pause(3)
    plt.close()
        
if __name__ == '__main__':
    starting_balance=100000
#    train(starting_balance, window_size=7, resume_epoch=0, max_epoch = 5000)      
    test(starting_balance, window_size=7, model_epi='2900')