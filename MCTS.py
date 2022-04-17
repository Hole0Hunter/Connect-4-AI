import copy
import random
from math import *
import sys

class Frame:
    
    def __init__(self, frame = None, rows = 6, columns = 5):
        if frame == None:
            frame = self.generate_empty_frame(rows, columns)
        self.frame = frame
        self.player = self.get_player()
        self.max_rows = len(frame)
        self.max_columns = len(frame[0])
    
    def generate_empty_frame(self, rows, columns):
        frame = []
        
        for row in range(rows):
            frame.append([0]*columns)
        
        return frame

    def print_frame(self):
        for row in self.frame:
            for cell in row:
                print(cell, end="  ")
            print()
        print() 
           
    def unroll_frame(self, frame = None):
        if frame == None:
            frame = self.frame
        unrolled_frame = ""
        for row in frame:
            for cell in row:
                unrolled_frame += str(cell)
        
        for index in range(len(unrolled_frame)):
            if unrolled_frame[index] == '0':
                index += 1
            else:
                break
        
        if index == len(unrolled_frame):
            return '0'
        
        return unrolled_frame[index:]
           
    def get_player(self):
        p1 = p2 = 0
        for row in self.frame:
            p1 += row.count(1)
            p2 += row.count(2)
        
        if p1 == p2:
            return 1
        if p1 == p2 + 1:
            return 2
        return -1
        
    def get_valid_moves(self):
        valid_moves = []
        
        for column in range(self.max_columns):
            if self.frame[0][column] == 0:
                valid_moves.append(column+1)
        
        return valid_moves

    def get_random_move(self):
        valid_moves = self.get_valid_moves()
        
        if len(valid_moves):
            return random.choice(valid_moves)
        
        return -1
    
    def play_move(self, move):
        next_state = copy.deepcopy(self)
        move -= 1
        
        for row in range(self.max_rows - 1, -1, -1):
            if self.frame[row][move] == 0:
                next_state.frame[row][move] = self.player
                if self.player == 1:
                    next_state.player = 2
                else:
                    next_state.player = 1
                return next_state
        
        return None

    def is_ended(self):
        for cell in self.frame[0]:
            if cell == 0:
                return False
        return True
    
    def get_winner(self):
        
        #row wise
        for row in self.frame:
            count = 1
            prev = 0
            for cell in row:
                if cell != 0 and cell == prev:
                    count += 1
                    if count == 4:
                        return cell
                else:
                    count = 1
                prev = cell

        #column wise
        for column in range(self.max_columns):
            count = 1
            prev = 0
            for row in range(self.max_rows):
                cell = self.frame[row][column]
                if cell != 0 and cell == prev:
                    count += 1
                    if count == 4:
                        return cell
                else:
                    count = 1
                prev = cell
        
        #main diagonal directional diagnals
        for row_start in range(self.max_rows):
            row = row_start
            column = 0
            count = 1
            prev = 0
            while row >= 0 and column < self.max_columns:
                cell = self.frame[row][column]
                if cell != 0 and cell == prev:
                    count += 1
                    if count == 4:
                        return cell
                else:
                    count = 1
                prev = cell
                row -= 1
                column += 1
        
        for column_start in range(1, self.max_columns):
            row = self.max_rows - 1
            column = column_start
            count = 1
            prev = 0
            while row >= 0 and column < self.max_columns:
                cell = self.frame[row][column]
                if cell != 0 and cell == prev:
                    count += 1
                    if count == 4:
                        return cell
                else:
                    count = 1
                prev = cell
                row -= 1
                column += 1
                
        #secondary diagonal directional diagnals
        for row_start in range(self.max_rows):
            row = row_start
            column = 0
            count = 1
            prev = 0
            while row < self.max_rows and column < self.max_columns:
                cell = self.frame[row][column]
                if cell != 0 and cell == prev:
                    count += 1
                    if count == 4:
                        return cell
                else:
                    count = 1
                prev = cell
                row += 1
                column += 1
        
        for column_start in range(1, self.max_columns):
            row = 0
            column = column_start
            count = 1
            prev = 0
            while row < self.max_rows and column < self.max_columns:
                cell = self.frame[row][column]
                if cell != 0 and cell == prev:
                    count += 1
                    if count == 4:
                        return cell
                else:
                    count = 1
                prev = cell
                row += 1
                column += 1
        
        #if no 4-in-a-row
        return 0     

class Game_Manager:
    
    def __init__(self):
        self.frame = Frame()
        print("Connect 4!")
        self.display_board()
    
    def player_move(self):
        move = int(input())
        
        try:
            return self.play_move(move)
        except:
            print("Invalid Move!")
            self.player_move()
    
    
    def MC200_move(self):
        move = MC200.get_move(self.frame)
        return self.play_move(move)
    
    def MC40_move(self):
        move = MC40.get_move(self.frame)
        return self.play_move(move)
    
    def play_move(self, move):
        self.print_move(move)
        self.frame = self.frame.play_move(move)
        return self.get_game_status()

    def print_move(self, move):
        print("Player", self.frame.get_player(), "plays", move)
        
    def get_game_status(self):
        self.display_board()
        return self.is_game_ended()
    
    def is_game_ended(self):
        return self.is_game_won() or self.is_board_full()
        
    def is_board_full(self):
        if self.frame.is_ended():
            print("DRAW")
            return True
        return False
    
    def is_game_won(self):
        winner = self.frame.get_winner()
        
        if winner > 0:
            if winner == 1:
                print("MC200 Won")
            else:
                print("MC40 won")
            return True
        
        return False

    def display_board(self):
        self.frame.print_frame()
        
class MC40:
    
    def get_move(frame: Frame):
        
        iterations = 40
        exploratory_factor = 1.5
        
        MC40 = MonteCarloTreeSearch(frame, exploratory_factor)
        action = MC40.monte_carlo_tree_search(iterations)
        
        return action
        
class MC200:
    
    def get_move(frame: Frame):
        
        iterations = 200
        exploratory_factor = 5
        
        MC200 = MonteCarloTreeSearch(frame, exploratory_factor)
        action = MC200.monte_carlo_tree_search(iterations)
        
        return action

class MonteCarloTreeSearch:
    score = { 'draw': 1, 'win': 1, 'lose': 0 } 
    
    def __init__(self, frame, exploratory_factor):
        self.frame : Frame = frame
        self.root : Node = Node(frame)
        self.exploratory_factor = exploratory_factor
        
    def monte_carlo_tree_search(self, iterations):

        #self.debug_frame()
        for iteration in range(iterations):
            if self.root.winner > 0 or self.root.ended:
                return -1
            self.iterate()
            #self.debug(iteration)
        
        action = self.get_best_move()
        return action

    def get_best_move(self):
        if self.root.winner > 0 or self.root.ended:
            return -1
        
        result = -1
        action = -1
        children = self.root.get_children_list()
        
        for child in children:
            current_average_wins = child.get_average_wins()
            if current_average_wins > result:
                result = current_average_wins
                action = child.action
                
        return action
       
    def iterate(self):
        
        if not self.root.has_children():
            self.expand(self.root)
        
        current = self.root
        
        while current.has_children():
            current = self.select(current)
        
        if current.winner > 0:
            if current.winner == self.root.player:
                self.backpropagate(current, MonteCarloTreeSearch.score['win'])
            else:
                self.backpropagate(current, MonteCarloTreeSearch.score['lose'])
            return

        if current.ended:
            self.backpropagate(current, MonteCarloTreeSearch.score['draw'])
            return
        
        if current.simulations:
            self.expand(current)
            current = self.select(current)
        
        winner = self.simulate(current)
        
        if winner == self.root.player:
            simulation_score = MonteCarloTreeSearch.score['win']
        elif winner <= 0:
            simulation_score = MonteCarloTreeSearch.score['draw']
        else:
            simulation_score = MonteCarloTreeSearch.score['lose']
            
        self.backpropagate(current, simulation_score)
    
    def select(self, node):
        children = node.get_children_list()
        ucb_score = {}
        selected_child, max_ucb_score = None, 0
        
        for child in children:
            ucb_score[child.action] = child.get_UCB(self.exploratory_factor)
            if ucb_score[child.action] == None:
                return child
            if ucb_score[child.action] > max_ucb_score:
                max_ucb_score = ucb_score[child.action]
                selected_child = child
        
        return selected_child
        
    def expand(self, node):
        
        base_frame = node.frame
        
        winner = base_frame.get_winner()
        if winner > 0:
            node.winner = winner
            return
            
        valid_moves = base_frame.get_valid_moves()
        if len(valid_moves) == 0:
            node.ended = True
            return
        
        for move in valid_moves:
            child = Node(base_frame.play_move(move), node, move)
            node.add_child(child)
    
    def simulate(self, node):
        #returns winner or 0 if drawn
        frame: Frame = node.frame
        
        while True:
            winner = frame.get_winner()
            if winner > 0:
                return winner
            move = frame.get_random_move()
            if move == -1:
                return 0
            frame = frame.play_move(move)
    
    def backpropagate(self, node, score):
        while True:
            node.simulations += 1
            node.win_counter += score
            
            if node == self.root:
                return
            node = node.parent
 
class Node:
    
    def __init__(self, frame: Frame, parent = None, action = -1):
        self.frame = frame
        self.parent = parent
        self.action = action
        self.ended = frame.is_ended()
        self.winner = frame.get_winner()
        self.player = frame.get_player()
        self.simulations = 0
        self.win_counter = 0
        self.children = {}
        
    def has_children(self):
        return len(self.children) > 0
     
    def get_average_wins(self):
        wins = self.win_counter
        node_simulations = self.simulations
        if node_simulations == 0:
            return 0
        
        return wins / node_simulations   
    
    def get_UCB(self, exploratory_factor):
        wins = self.win_counter
        node_simulations = self.simulations
        
        if self.parent == None:
            return None
        
        total_simulations = self.parent.simulations
        
        if node_simulations == 0:
            return None
        
        trust_value = wins / node_simulations
        exploratory_factor = sqrt((exploratory_factor * log2(total_simulations)) / node_simulations)
        
        ucb = trust_value + exploratory_factor
        
        return ucb

    def get_child_by_action(self, action):
        return self.children.get(action, None)

    def get_children_list(self):
        return list(self.children.values())


    def set_children(self, children):
        for child in children:
            self.children[child.action] = child
    
    def add_child(self, child):
        self.children[child.action] = child

  
##############

#Your program can go here.

###############

def MCX_vs_MCX():
    game = Game_Manager()
    while True:
        if game.MC200_move() or game.MC40_move():
            return

if __name__ == "__main__":
    print("Give option:")
    print("1- MC200 vs MC40")
    print("2- Mc200 vs Q learning")
    choice = int(input())
    if choice == 1:
        game = Game_Manager()
        while True:
            if game.MC200_move() or game.MC40_move():
                break
        

# def PrintGrid(positions):
#     print('\n'.join(' '.join(str(x) for x in row) for row in positions))
#     print()

# def main():
    
    
#     print("************ Sample output of your program *******")

#     game1 = [[0,0,0,0,0],
#           [0,0,0,0,0],
#           [0,0,1,0,0],
#           [0,2,2,0,0],
#           [1,1,2,2,0],
#           [2,1,1,1,2],
#         ]


#     game2 = [[0,0,0,0,0],
#           [0,0,0,0,0],
#           [0,0,1,0,0],
#           [1,2,2,0,0],
#           [1,1,2,2,0],
#           [2,1,1,1,2],
#         ]

    
#     game3 = [ [0,0,0,0,0],
#               [0,0,0,0,0],
#               [0,2,1,0,0],
#               [1,2,2,0,0],
#               [1,1,2,2,0],
#               [2,1,1,1,2],
#             ]

#     print('Player 2 (Q-learning)')
#     print('Action selected : 2')
#     print('Value of next state according to Q-learning : .7312')
#     PrintGrid(game1)


#     print('Player 1 (MCTS with 25 playouts')
#     print('Action selected : 1')
#     print('Total playouts for next state: 5')
#     print('Value of next state according to MCTS : .1231')
#     PrintGrid(game2)

#     print('Player 2 (Q-learning)')
#     print('Action selected : 2')
#     print('Value of next state : 1')
#     PrintGrid(game3)
    
#     print('Player 2 has WON. Total moves = 14.')
    
# if __name__=='__main__':
#     main()