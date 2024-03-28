# 1차 논문 검증을 위한 교육용 Vanila PPO 예제 입니다.
# 결과에 대한 신뢰성은 절대로 담보할 수 없으니 공부하시는 용도로만 사용하세요.
#  
# 코드는 데이터 파일 (SS00001.csv)하고 같은 폴더에서 그냥 돌리면 됩니다.
# 테스트는 맨 아래 메인에서 comment한 부분 바꿔 주시면 됩니다.
#
# State = [sigmoid(과거 7 일의 종가의 차이) + log(포트폴리오(현금+주식), 홀딩, 종가) + 4가지 기술적지표]
# reward = 포트폴리오의 변화
# Action = -10 ~ 10 
#

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
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

train_data = data.loc[(data.index > '1998-01-01') & (data.index <= '2019-11-31'), :]
test_data = data.loc[(data.index >= '2019-12-01') & (data.index <= '2021-05-31'), :]

# Critic에 Generalized Advantage Estimation (GAE) 를 사용했습니다.

buffer_length = 5000
#Hyperparameters
learning_rate = 1.0e-6 
gamma         = 0.98
epsilon       = 0.10  #PPO clip para
beta          = 0.01  #PPO entropy para
lam           = 0.5   # if lam = 0 A2C is same as TD AC
val_loss_coef = 0.5   #Critic loss contribution
batch_size    = 128
max_trade_share = 10  # 최대 buy or sell position
action_space  = 2*max_trade_share + 1

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
        self.total_episodes = len(data)
        self.cur_step = 0
        self.mode = mode

    def reset(self):
        self.cash = self.starting_balance
        self.shares = 0
        if self.mode == 'train':
            self.cur_step = self.next_episode
        else:
            self.cur_step = 0
        return self.next_observation()

    
    def step(self, action):   # assume every day 
        balance = self.cur_balance
        self.cur_step += 1
        if self.cur_step < self.total_steps:
            self.take_action(action)
            state = self.next_observation()
            reward = self.cur_balance - balance
        
        done = self.cur_step == self.total_steps - 1
        return state, reward, done
    
    # 한루에 한번 씩 사거나 파는 것으로 가정
    def take_action(self, actions): 
        action = actions - 10
        if action > 0:
            share = action
            price = self.cur_close_price * (1 + self.commission_rate)
            if self.cash < price * share:
                share = int(self.cash / (price * share))
            self.cash -= price * share
            self.shares += share

        elif action < 0:
            share = -1*action
            price = self.cur_close_price * (1 - self.commission_rate)
            if self.shares < share:
                share = self.shares
            self.cash += price * share
            self.shares -= share

 
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

class ReplayBuffer():
    def __init__(self, buffer_len = 1000):
        self.buffer = deque(maxlen = buffer_len)
    
    def put_data(self, transition):
        self.buffer.append(transition)
    
    def make_batch(self):
        s_lst, a_lst, r_lst, p_a_lst, s_prime_lst, done_lst = [], [], [], [], [], []
        start =  random.randrange(0, self.size()-batch_size)
        mini_batch = [self.buffer[i] for i in range(start,start+batch_size)]
        for transition in mini_batch:
            s, a, r, p_a, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            p_a_lst.append([p_a])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_lst = np.array(s_lst)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        p_a_lst = np.array(p_a_lst)
        s_prime_lst = np.array(s_prime_lst)
        done_lst = np.array(done_lst)

        s_batch, a_batch, r_batch, p_a_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), \
                                                            torch.tensor(a_lst), \
                                                            torch.tensor(r_lst, dtype=torch.float), \
                                                            torch.tensor(p_a_lst, dtype=torch.float), \
                                                            torch.tensor(s_prime_lst, dtype=torch.float), \
                                                            torch.tensor(done_lst, dtype=torch.float)

        return s_batch, a_batch, r_batch, p_a_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)  

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()

        self.fc1_pi = nn.Linear(state_size, 512)  
        self.fc2_pi = nn.Linear(512, 256)  
        self.fc3_pi = nn.Linear(256, action_size) 
        self.fc1_v = nn.Linear(state_size, 512) 
        self.fc2_v = nn.Linear(512, 256)
        self.fc3_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim = -1):
        x = F.tanh(self.fc1_pi(x))
        x = F.tanh(self.fc2_pi(x))
    #    prob = F.tanh(self.fc3_pi(x))
        prob = F.softmax(self.fc3_pi(x), dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.tanh(self.fc1_v(x))
        x = F.tanh(self.fc2_v(x))
        v = self.fc3_v(x)
        return v


def train_net(model, memory):
    total_loss = []
    model.train()

    for _ in range(4):
        s, a, r, prob_a, s_prime, done = memory.make_batch()
        v_cur = model.v(s)
        v_next = model.v(s_prime)

        # generalized advanced estimation GAE
        advs = torch.zeros_like(r)
        future_ages = torch.tensor(0.0, dtype = r.dtype)

        for t in reversed(range(batch_size)):
            delta = r[t] + gamma * v_next[t] * done[t] - v_cur[t]
            advs[t] = future_ages = delta + gamma * lam * done[t] * future_ages 
              
        v_target = advs + v_cur
        
        # standadize advs
        advs = (advs - advs.mean())/(advs.std() + 1.0e-8)
        
        # current policy
        pi_cur = model.pi(s, softmax_dim=-1)  # from new policy
        pi_a = pi_cur.gather(1,a.to(torch.long))  # need int64

        # PPO
        ratios = torch.exp(torch.log(pi_a)-torch.log(prob_a))
        sur_1 = ratios * advs    # Eq 7.17
        sur_2 = torch.clamp(ratios, 1.0-epsilon, 1.0+epsilon) * advs
        policy_loss = -torch.min(sur_1, sur_2).mean()   #Eq 7.34

        # entropy regularization
        entropy = -torch.sum(torch.abs(pi_cur) * torch.log(torch.abs(pi_cur))) / np.log(len(pi_cur))

        ent_penalty = -beta * entropy
        val_loss = val_loss_coef * F.mse_loss(v_cur, v_target)
        loss = policy_loss + ent_penalty + val_loss

        total_loss.append(loss.detach().numpy())
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()   

    memory.buffer = []  # clear the history since it is on-policy
    return np.mean(total_loss)

def train(window_size=20, starting_balance = 1000000, resume_epoch=0, max_epoch=1000):  
    mode = 'train'
    env = Trade(train_data, starting_balance, window_size, mode)
    state_size = window_size + 7
    action_size = action_space
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    model = ActorCritic(state_size, action_size)

    memory = ReplayBuffer()

    save_interval = 100
    epochs = max_epoch
    loss_history = []
    pv_history = []    # portfolio history

    # to continue from the previous saving

    start_epoch = resume_epoch
    
    if start_epoch > 0:
        model.load_state_dict(torch.load("PPOmodel_ep" + str(start_epoch)))
        
    pbar = tqdm(range(start_epoch, epochs))

    for n_epi in pbar:
        s = env.reset()
        done = False
        action_history = []

        # complete one episode

        while not done:
            s = torch.from_numpy(s).float()
            prob = model.pi(s, softmax_dim=-1)  #state1: torch.size([7])
            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done = env.step(a)
            memory.put_data((s, a, r, prob[a].item(), s_prime, done))
            action_history.append( a )    
            s = s_prime

        np_actions = np.array(action_history)
        index_0 = len(np.where(np_actions == 0)[0])
        index_1 = len(np.where(np_actions > 0)[0])
        index_2 = len(np.where(np_actions < 0)[0])
        pv_history.append(env.cur_balance)
        pbar.set_description(str(index_0)+"/"+str(index_1)+"/"+str(index_2)+"/"+"%.4f" % env.cur_balance)

        if memory.size() > batch_size:        
            loss = train_net(model, memory)
            loss_history.append(loss)
        
        # save log every n step
        if n_epi % save_interval == 0:
            torch.save(model.state_dict(), "PPOmodel_ep" + str(n_epi))    

    torch.save(model.state_dict(), "PPOmodel_epfinal") 

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex = True)

    axs[0].plot(loss_history)
    axs[0].set_ylabel('loss', fontsize=12)
    axs[0].set_xlabel("date",fontsize=12)

    axs[1].plot(pv_history)
    axs[1].set_ylabel('pv', fontsize=12)
    axs[1].set_xlabel("date",fontsize=12)

    plt.savefig('PPO_S00001_train.png')
    plt.show()
    plt.pause(3)
    plt.close()
    
def test(window_size = 20, starting_balance = 1000000, model_epi = 'final'):  
    mode = 'test'
    env = Trade(test_data, starting_balance, window_size, mode)
    state_size = window_size + 7
    action_size = action_space
    model = ActorCritic(state_size, action_size)

    model.load_state_dict(torch.load("PPOmodel_ep" + str(model_epi)))
    
    model.eval()

    action_history = []
    pv_history = []

    s = env.reset()
    done = False

    # start from any random position of the training data
    while not done:
        state = torch.from_numpy(s).float()
        prob = model.pi(state)  #state1: torch.size([7])
        s_prime, r, done = env.step(prob)
        s = s_prime 
        action_history.append( int(prob.item()*max_trade_share) )    
        pv = np.exp(s_prime[window_size])    
        pv_history.append(pv)                  

    np_actions = np.array(action_history)
    test_close = test_data["Close"].values

    index_0 = np.where(np_actions == 0)[0]
    index_1 = np.where(np_actions > 0)[0]
    index_2 = np.where(np_actions < 0)[0]
    print(str(len(index_0))+"/"+str(len(index_1))+"/"+str(len(index_2))+"/"+"%.4f" % env.cur_balance)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex = True)    

#    axs[0].scatter(index_0, test_close[index_0], c='red', label='hold', marker='^')
    if len(index_0) > 0:
        axs[0].scatter(index_0, test_close[index_0], c='red', label='hold', marker='^')
    if len(index_1) > 0:
        axs[0].scatter(index_1, test_close[index_1], c='green', label='buy', marker='>')
    if len(index_2) > 0:
        axs[0].scatter(index_2, test_close[index_2], c='blue', label='sell', marker='v')
    axs[0].plot(test_close)
    axs[0].legend()
    axs[0].set_ylabel('Close', fontsize=22)

    axs[1].plot(pv_history, c='red', label='pv')
    axs[1].plot(test_data['Close'].values * starting_balance / test_data['Close'].iloc[0], c='black', label='close')
    axs[1].legend()

    axs[1].set_ylabel('Portfolio', fontsize=22)
    axs[1].set_xlabel("date",fontsize=22)
    plt.savefig('PPO_S00001_test.png')
    plt.show()
    plt.pause(3)
    plt.close()
        
        
if __name__ == '__main__':
    starting_balance=100000
    train(window_size = 7, starting_balance=starting_balance, resume_epoch = 0, max_epoch = 1000)      
#    test(window_size = 7, starting_balance=starting_balance, model_epi = 'final')
