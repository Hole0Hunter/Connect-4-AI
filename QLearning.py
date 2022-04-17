from Frame import Frame
from MC200 import MC200
import random
import pandas as pd


class QLearning:
    reward = {'win': 999, 'lose': -999, 'draw': 400, 'move': -1}
    
    def __init__(self, gamma = 0.9, alpha = 0.2, epsilon = 0.8):
        self.frame : Frame = Frame(rows = 2)
        self.gamma = gamma
        self.alpha = alpha
        self.player = self.frame.get_player()
        self.epsilon = epsilon
        self.q_table = {}
    
    def epsilon_greedy(self, key):
        moves = self.q_table[key]
        prob = random.random()
        if prob < self.epsilon:
            return self.greedy(moves)

        return random.choice(list(moves.keys()))
        
    def greedy(self, moves):
        key = None
        for move in moves:
            if key == None or moves[move] > moves[key]:
                key = move
        
        return key
    
    def get_move(self):
        self.train()
        action = None
        state = self.frame.unroll_frame()
        for key in self.q_table[state]:
            if action == None or self.q_table[state][action] >= self.q_table[state][key]:
                action = key
        
        return action
    
    def learn(self):
        self.train()
        self.store_data()
    
    def store_data(self):
        # nested dictionary to pandas data frame
        
        ids = []
        ndict = []

        for user_id in self.q_table:
            ids.append(user_id)
            ndict.append(pd.DataFrame.from_dict(self.q_table[user_id], orient='index'))

        help = pd.concat(ndict, keys=ids)

        help.to_csv('./QL/2019A7PS0032G_Mohit.dat')
    
    def read_best_move(self):
        input_df = pd.read_table('2019A7PS0032G_Mohit.dat') 
        action = None
        state = self.frame.unroll_frame()
        for key in input_df[state]:
            if action == None or input_df[state][action] >= input_df[state][key]:
                action = key
        
        return action
        
    
    def get_reward(self, state: Frame, action):
        next_state = state.play_move(action)
        winner = next_state.get_winner()
        
        if winner == self.player:
            reward = QLearning.reward['win']
        elif winner > 0:
            reward = QLearning.reward['lose']
        elif next_state.is_ended():
            reward = QLearning.reward['draw']
        else:
            reward = QLearning.reward['move']
        
        return reward

    def get_actions(self, state: Frame):
        return state.get_valid_moves()
    
    def init_q_table(self, state):
        key = state.unroll_frame()
            
        if not key in self.q_table:
            self.q_table[key] = {}
            actions = state.get_valid_moves()
            
            for action in actions:
                self.q_table[key][action] = 0
        
        return key
    
    def train(self):
        
        iterations = 10000
        
        for iteration in range(iterations):
            current_state = self.frame
            action = MC200.get_move(current_state)
            current_state = current_state.play_move(action)
            
            while current_state.get_winner() <= 0 or not current_state.is_ended():
                
                current_key = self.init_q_table(current_state)
                if len(self.q_table[current_key]) == 0:
                    break
                action = self.epsilon_greedy(current_key)
                next_state = current_state.play_move(action)
  
                
                next_key = self.init_q_table(next_state)
                
                key_q_max = None
                for key in self.q_table[next_key]:
                    if key_q_max == None or self.q_table[next_key][key] > self.q_table[next_key][key_q_max]:
                        key_q_max = key
        
                next_reward = self.get_reward(self.frame, action)
                if next_reward == QLearning.reward['move'] and key_q_max != None and action != None:
                    TD = next_reward + self.gamma * self.q_table[next_key][key_q_max] - self.q_table[current_key][action]
                else:
                    TD = next_reward
                    self.q_table[current_key][action] += self.alpha * TD
                    break
                
                self.q_table[current_key][action] += self.alpha * TD
                
                move = MC200.get_move(next_state)
                if current_state.get_winner() or current_state.is_ended() or next_state.get_winner() or current_state.is_ended():
                    break
                current_state = next_state.play_move(move)
                if current_state.get_winner() or current_state.is_ended():
                    break

if __name__ == "__main__":
    q = QLearning()
    q.train()
    q.store_data()
    
            