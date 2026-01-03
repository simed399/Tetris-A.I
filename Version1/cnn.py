import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import argparse
from tetris import Tetris
from collections import deque
import wandb


class SimpleCNN(nn.Module):
    """Simple CNN for Tetris board state evaluation"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 20x10 board (1 channel)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # After pooling: 5x2 with 64 channels = 640
        self.fc1 = nn.Linear(64 * 5 * 2, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # x shape: (batch, 1, 20, 10)
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # -> (batch, 32, 10, 5)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # -> (batch, 64, 5, 2)
        
        x = x.view(-1, 64 * 5 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNAgent:
    def __init__(self, lr=0.001, gamma=0.99):
        self.model = SimpleCNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Load model if exists
        self.model_path = "model/cnn_model.pth"
        self.load_model()
        
    def load_model(self):
        if os.path.isfile(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
            print(f'CNN Model Loaded from {self.model_path}')
        else:
            print("No saved CNN model found. Using a new model.")
    
    def save_model(self):
        os.makedirs("model", exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, self.model_path)
        print("CNN Model saved successfully!")
    
    def board_to_tensor(self, board):
        """Convert board to tensor format (1, 1, 20, 10)"""
        board_binary = (board > 0).astype(np.float32)
        return torch.FloatTensor(board_binary).unsqueeze(0).unsqueeze(0)
    
    def get_action(self, tetris):
        """Get best action using CNN"""
        next_states = tetris.game.calc_all_states()
        
        if not next_states:
            return None
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(list(next_states.keys()))
        
        # Exploitation: evaluate all possible states
        best_action = None
        best_value = float('-inf')
        
        for action, state_features in next_states.items():
            # Create temporary board state
            board = self._get_board_after_action(tetris, action)
            board_tensor = self.board_to_tensor(board)
            
            with torch.no_grad():
                value = self.model(board_tensor).item()
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def _get_board_after_action(self, tetris, action):
        """Simulate board state after action"""
        board = tetris.game.board.copy()
        x, rotations = action
        
        # Simple simulation - just mark the landing position
        piece = tetris.game.tetromino
        for r in range(int(rotations)):
            piece.rotate()
        
        if piece.move_to_x(x):
            temp_board = (board != 0).astype(int)
            piece.hard_drop(temp_board, False)
            
            # Reset piece position
            piece.move_to_y(0)
            for _ in range(int(rotations)):
                piece.rotate()
            piece.move_to_x(4)
            
            return temp_board
        
        return board
    
    def remember(self, state_board, reward, next_state_board, done):
        """Store experience in memory"""
        self.memory.append((state_board, reward, next_state_board, done))
    
    def train(self):
        """Train the CNN on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        
        total_loss = 0
        for state_board, reward, next_state_board, done in batch:
            state_tensor = self.board_to_tensor(state_board)
            
            current_q = self.model(state_tensor)
            
            if done:
                target_q = torch.FloatTensor([[reward]])
            else:
                next_tensor = self.board_to_tensor(next_state_board)
                with torch.no_grad():
                    next_q = self.model(next_tensor)
                target_q = torch.FloatTensor([[reward + self.gamma * next_q.item()]])
            
            loss = self.criterion(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / self.batch_size
    
    def calculate_reward(self, tetris):
        """Simple reward calculation"""
        game = tetris.game
        reward = 0
        
        # Line clears
        if game.lines_removed > 0:
            reward += game.lines_removed ** 2 * 100
        
        # Game over penalty
        if game.finished:
            reward -= 500
        else:
            reward += 1  # Survival bonus
        
        return reward


def train_cnn(slow_drop=True, games=1000):
    """Train the CNN agent"""
    # Initialize W&B
    wandb.init(
        project="tetris-cnn",
        config={
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "batch_size": 32,
            "memory_size": 10000,
            "games": games,
            "slow_drop": slow_drop
        },
        name=f"cnn_slowdrop_{slow_drop}_{games}games"
    )
    
    agent = CNNAgent()
    tetris = Tetris(i=0, SLOW_DROP=slow_drop)
    
    scores = []
    
    for game_num in range(1, games + 1):
        tetris.reset()
        done = False
        total_reward = 0
        moves = 0
        
        state_board = tetris.game.board.copy()
        
        while not done and moves < 500:
            action = agent.get_action(tetris)
            
            if action is None:
                break
            
            # Execute action
            tetris.play_full(action)
            
            next_state_board = tetris.game.board.copy()
            reward = agent.calculate_reward(tetris)
            done = tetris.game.finished
            
            # Remember experience
            agent.remember(state_board, reward, next_state_board, done)
            
            state_board = next_state_board
            total_reward += reward
            moves += 1
        
        # Train after each game
        loss = agent.train()
        
        scores.append(tetris.game.score)
        
        # Log to W&B
        wandb.log({
            "game": game_num,
            "score": tetris.game.score,
            "lines": tetris.game.lines,
            "epsilon": agent.epsilon,
            "loss": loss,
            "total_reward": total_reward,
            "moves": moves,
            "avg_score_10": np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores),
            "avg_score_100": np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        })
        
        if game_num % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(f'Game {game_num}/{games} | Score: {tetris.game.score} | '
                  f'Lines: {tetris.game.lines} | Avg Score: {avg_score:.1f} | '
                  f'Epsilon: {agent.epsilon:.3f} | Loss: {loss:.4f}')
        
        if game_num % 100 == 0:
            agent.save_model()
    
    agent.save_model()
    wandb.finish()
    print(f'\nTraining completed! Final avg score: {np.mean(scores[-100:]):.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Tetris AI with CNN')
    parser.add_argument('--slow_drop', type=str, default='true', 
                        help='Use slow drop mode (true/false)')
    parser.add_argument('--games', type=int, default=1000,
                        help='Number of games to train')
    
    args = parser.parse_args()
    slow_drop = args.slow_drop.lower() == 'true'
    
    print(f'Starting CNN training with slow_drop={slow_drop}, games={args.games}')
    train_cnn(slow_drop=slow_drop, games=args.games)
