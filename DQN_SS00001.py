# 1차 논문 검증을 위한 교육용 Native DQN 예제 입니다.
# 결과에 대한 신뢰성은 절대로 담보할 수 없으니 공부하시는 용도로만 사용하세요.
#  
# 코드는 데이터 파일 (SS00001.csv)하고 같은 폴더에서 그냥 돌리면 됩니다.
# 테스트는 맨 아래 메인에서 comment한 부분 바꿔 주시면 됩니다.
#
# State = [sigmoid(과거 7 일의 종가의 차이) + log(포트폴리오(현금+주식), 홀딩, 종가) + 4가지 기술적지표]
# reward = 포트폴리오의 변화
# Action = -10 to 10 // 21 dimension
#

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

train_data = data.loc[(data.index > '1998-01-01') & (data.index <= '2019-11-31'), :]
test_data = data.loc[(data.index >= '2019-12-01') & (data.index <= '2021-05-31'), :]

# Generalized Advantage Estimation (GAE) and n-step Return

#Hyperparameters
learning_rate = 1.0e-6
gamma         = 0.98
buffer_limit  = 1000
batch_size    = 32     # for mini_batch
max_trade_share = 10
tau           = 0.001
action_space  = 2*max_trade_share + 1

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Trade():
    def __init__(self, data, starting_balance=1000000, episodic_length=20, mode='train'):  
        super(Trade, self).__init__()
        self.data = data
        self.price_column = 'Close'
        self.dClose_column = 'dClose'
        self.indicator_columns = ['RSI', 'MACD', 'CCI', 'ADX']
        self.episodic_length = episodic_length
        self.starting_balance = starting_balance
        self.commission_rate = 0.0
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
        if self.cur_step < self.total_steps - 1:
            self.take_action(action)
            state = self.next_observation()
            reward = self.cur_balance - balance
        
        done = self.cur_step == self.total_steps - 2
        return state, reward, done
    
    def take_action(self, action):  
        action -= max_trade_share
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
        return random.randrange(0, self.total_episodes - batch_size)

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
    def __init__(self):
        self.buffer = deque(maxlen = buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for transition in mini_batch:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])
        
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)  

class Dqn(nn.Module):
    def __init__(self, state_size, action_size):
        super(Dqn, self).__init__()
     
        self.fc1 = nn.Linear(state_size, 512)  
        self.ln1   = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)  
        self.ln2   = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, action_size) 

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        value = self.fc3(x)
        return value

    def sample_action(self, state, epsilon):         
        out = self.forward(state)
        coin = random.random()
        if coin < epsilon:
            return np.random.randint(0, action_space)
        else:
            return out.argmax().item()   

def train_net(q, memory, optimizer):
    total_loss = []
    q.train()
    
        # we will randomly process n times of batches from the buffer 
    for _ in range(4):
        s, a, r, s_prime, done = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1,a)

        with torch.no_grad():
            qtarget_out = q(s_prime)
        bestaction = qtarget_out.argmax(1).unsqueeze(1) #qtarget_out.max(1)[1].unsqueeze(1)    #
        
        max_q_prime = qtarget_out.gather(1,bestaction) 
        target = r + gamma * max_q_prime * done
        loss = F.mse_loss(q_a, target)
        total_loss.append(loss.detach().numpy().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

    return np.mean(total_loss)

def train(window_size=20, starting_balance = 100000, resume_epoch=0, max_epoch=1000):  
    mode = 'train'
    env = Trade(train_data, starting_balance, window_size, mode)
    state_size = window_size + 7
    action_size = action_space
    q = Dqn(state_size, action_size)
    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    
    save_interval = 100
    epochs = max_epoch
    loss_history = []
    pv_history = []    # portfolio history
    # to continue from the previous saving

    start_epoch = resume_epoch

    torch.manual_seed(0)
    np.random.seed(0)

    if start_epoch > 0:
        q.load_state_dict(torch.load("Dqnmodel_ep" + str(start_epoch)))

    pbar = tqdm(range(start_epoch, epochs))

    for n_epi in pbar:
        if start_epoch == 0:
            epsilon = max(0.01, 1.0 - 1.0*n_epi/epochs)
        else:
            epsilon = 0.01

        s = env.reset()
        done = False
        action_history = []
        # complete one episode
 
#        while not done:
        for _ in range(2000):
            state = torch.from_numpy(s).float()
            a = q.sample_action(state, epsilon)
            s_prime, r, done = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime
            action_history.append(a)  
            if done:
                break
                                     
        if memory.size() > batch_size:
            loss = train_net(q, q_target, memory, optimizer)

        loss_history.append(loss)

        np_actions = np.array(action_history)
        index_0 = len(np.where(np_actions == 10)[0])
        index_1 = len(np.where(np_actions > 10)[0])
        index_2 = len(np.where(np_actions < 10)[0])
        pv_history.append(env.cur_balance)
        pbar.set_description(str(index_0)+"/"+str(index_1)+"/"+str(index_2)+"/"+"%.4f" % env.cur_balance)

        # save log every n step
        if n_epi % save_interval == 0:
            torch.save(q.state_dict(), "DDqnmodel_ep" + str(n_epi))    

    torch.save(q.state_dict(), "Dqnmodel_epfinal") 

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex = True)

    axs[0].plot(loss_history)
    axs[0].set_ylabel('policy_loss', fontsize=12)
    axs[0].set_xlabel("update",fontsize=12)

    axs[1].plot(pv_history)
    axs[1].set_ylabel('profit', fontsize=12)
    axs[1].set_xlabel("date",fontsize=12)

    plt.savefig('DQN_S00001_test.png')
    plt.show()
    plt.pause(3)
    plt.close()

    
def test(window_size = 20, starting_balance = 100000, model_epi = 'final'):  
    mode = 'test'
    env = Trade(test_data, starting_balance , window_size, mode)
    state_size = window_size + 7
    action_size = action_space
    q = Dqn(state_size, action_size)
    q.load_state_dict(torch.load("Dqnmodel_ep" + str(model_epi)))
    q.eval()

    action_history = []
    pv_history = []    # portfolio history

    s = env.reset()
    done = False
    
    # start from any random position of the training data
    while not done:
        state = torch.from_numpy(s).float()
        q_out = q(state)
        a = q_out.argmax().numpy().item() #
        s_prime, r, done = env.step(a)
        s = s_prime
        action_history.append(a)
        pv = np.exp(s_prime[window_size])    
        pv_history.append(pv)                     

    np_actions = np.array(action_history)
    test_close = test_data["Close"].values
    
    index_0 = np.where(np_actions == 10)[0]
    index_1 = np.where(np_actions > 10)[0]
    index_2 = np.where(np_actions < 10)[0]

    print(str(len(index_0))+"/"+str(len(index_1))+"/"+str(len(index_2))+"/"+"%.4f" % env.cur_balance)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex = True)
    
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
    plt.savefig('DQN_S00001_test.png')
    plt.show()
    plt.pause(3)
    plt.close()
        
if __name__ == '__main__':
    starting_balance = 100000 
#   train(window_size=7, starting_balance = starting_balance, resume_epoch=0, max_epoch = 1000)      
    test(window_size=7, starting_balance=starting_balance, model_epi='final')