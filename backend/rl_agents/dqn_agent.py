# dqn_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from typing import Tuple, List, Dict, Any, Optional
from collections import deque, namedtuple
import pickle

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNetwork(nn.Module):
    """
    Deep Q-Network architecture for maze solving.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 128):
        super(DQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    """
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Network Agent for solving maze problems.
    """
    
    def __init__(self, state_size: Tuple[int, int], action_size: int = 4,
                 learning_rate: float = 0.001, discount_factor: float = 0.99,
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995,
                 min_exploration_rate: float = 0.01, seed: Optional[int] = None,
                 buffer_size: int = 10000, batch_size: int = 64,
                 target_update_freq: int = 100, hidden_size: int = 128,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        # Environment parameters
        self.state_size = state_size
        self.action_size = action_size
        
        # Learning parameters
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate
        
        # DQN specific parameters
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        
        # Set random seeds
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Calculate input dimension for neural network
        # We'll use a flattened representation of the maze + agent position
        self.input_dim = state_size[0] * state_size[1] + 2  # +2 for agent position
        
        # Initialize neural networks
        self.q_network = DQNetwork(self.input_dim, action_size, hidden_size).to(self.device)
        self.target_network = DQNetwork(self.input_dim, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_steps = []
        self.losses = []
        
        # State tracking
        self.state_visits = np.zeros(state_size)
        
        # Store maze for state preprocessing
        self.current_maze = None
    
    def state_to_tensor(self, state: Tuple[int, int], maze: np.ndarray) -> torch.Tensor:
        """
        Convert state (position) and maze to tensor for neural network.
        
        Args:
            state: Current position (row, col)
            maze: Current maze layout
            
        Returns:
            Tensor representation of state
        """
        # Flatten maze
        maze_flat = maze.flatten()
        
        # Create position vector
        position = np.array(state)
        
        # Concatenate maze and position
        state_vector = np.concatenate([maze_flat, position])
        
        # Convert to tensor
        return torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
    
    def choose_action(self, state: Tuple[int, int], maze: Optional[np.ndarray] = None) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current position (row, col)
            maze: Current maze layout
            
        Returns:
            Selected action
        """
        # Use stored maze if not provided
        if maze is None:
            maze = self.current_maze
        
        # Update state visits
        self.state_visits[state] += 1
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            # Exploitation: choose best action based on Q-values
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state, maze)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        else:
            # Exploration: random action
            return random.randint(0, self.action_size - 1)
    
    def store_experience(self, state: Tuple[int, int], action: int, 
                        reward: float, next_state: Tuple[int, int], 
                        done: bool, maze: np.ndarray):
        """
        Store experience in replay buffer.
        """
        experience = Experience(
            self.state_to_tensor(state, maze).cpu(),
            action,
            reward,
            self.state_to_tensor(next_state, maze).cpu(),
            done
        )
        self.memory.push(experience)
    
    def learn(self):
        """
        Sample from replay buffer and perform gradient descent.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        experiences = self.memory.sample(self.batch_size)
        
        # Separate batch components
        states = torch.cat([e.state for e in experiences]).to(self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float).to(self.device)
        next_states = torch.cat([e.next_state for e in experiences]).to(self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Store loss
        self.losses.append(loss.item())
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """
        Decay exploration rate.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def train(self, env, num_episodes: int = 1000, max_steps: int = 1000,
              verbose: bool = True, save_path: Optional[str] = None,
              save_interval: int = 100) -> Dict[str, List]:
        """
        Train the DQN agent.
        
        Args:
            env: Maze environment
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Print training progress
            save_path: Path to save models
            save_interval: Save model every N episodes
            
        Returns:
            Training history
        """
        for episode in range(num_episodes):
            state = env.reset()
            self.current_maze = env.maze.copy()
            episode_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Choose action
                action = self.choose_action(state, self.current_maze)
                
                # Take action
                next_state, reward, done, _ = env.step(action)
                
                # Store experience
                self.store_experience(state, action, reward, next_state, done, self.current_maze)
                
                # Learn from experience
                if len(self.memory) >= self.batch_size:
                    self.learn()
                
                # Update state
                state = next_state
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Store episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Print progress
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_loss = np.mean(self.losses[-1000:]) if self.losses else 0
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Reward: {episode_reward:.2f}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Steps: {steps}, "
                      f"Epsilon: {self.epsilon:.4f}, "
                      f"Loss: {avg_loss:.4f}")
            
            # Save model
            if save_path and (episode + 1) % save_interval == 0:
                self.save_model(f"{save_path}/dqn_episode_{episode + 1}.pth")
        
        # Save final model
        if save_path:
            self.save_model(f"{save_path}/dqn_final.pth")
        
        return {
            "rewards": self.episode_rewards,
            "steps": self.episode_steps,
            "losses": self.losses
        }
    
    def save_model(self, path: str):
        """
        Save model and training state.
        """
        # Nếu path là .pkl, chuyển sang .pth cho PyTorch
        if path.endswith('.pkl'):
            pth_path = path.replace('.pkl', '.pth')
            pkl_path = path
        else:
            pth_path = path
            pkl_path = path.replace('.pth', '.pkl')
        
        # Lưu PyTorch model
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'state_visits': self.state_visits,
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'losses': self.losses
        }, pth_path)
        
        # Lưu thông tin agent cho compatibility với evaluation.py
        agent_state = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'epsilon': self.epsilon,
            'state_visits': self.state_visits
        }
        
        # Lưu file .pkl để tương thích
        with open(pkl_path, 'wb') as f:
            pickle.dump(agent_state, f)
    
    def load_model(self, path: str):
        """
        Load model and training state.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.state_visits = checkpoint['state_visits']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_steps = checkpoint.get('episode_steps', [])
        self.losses = checkpoint.get('losses', [])
        
        # Set networks to appropriate mode
        self.q_network.eval()
        self.target_network.eval()
    
    def get_policy(self) -> np.ndarray:
        """
        Get policy matrix for visualization.
        """
        policy = np.zeros(self.state_size, dtype=int)
        
        # Create a simple maze for policy extraction
        if self.current_maze is None:
            # Create default maze (0 = path, 1 = wall)
            maze = np.zeros(self.state_size)
        else:
            maze = self.current_maze
        
        for r in range(self.state_size[0]):
            for c in range(self.state_size[1]):
                if maze[r, c] == 0:  # Only for valid positions
                    state = (r, c)
                    with torch.no_grad():
                        state_tensor = self.state_to_tensor(state, maze)
                        q_values = self.q_network(state_tensor)
                        policy[r, c] = q_values.argmax().item()
        
        return policy
    
    def get_value_function(self) -> np.ndarray:
        """
        Get value function for visualization.
        """
        values = np.zeros(self.state_size)
        
        # Create a simple maze for value extraction
        if self.current_maze is None:
            maze = np.zeros(self.state_size)
        else:
            maze = self.current_maze
        
        for r in range(self.state_size[0]):
            for c in range(self.state_size[1]):
                if maze[r, c] == 0:  # Only for valid positions
                    state = (r, c)
                    with torch.no_grad():
                        state_tensor = self.state_to_tensor(state, maze)
                        q_values = self.q_network(state_tensor)
                        values[r, c] = q_values.max().item()
        
        return values
    
    def visualize_policy(self, maze: np.ndarray):
        """
        Visualize the learned policy.
        """
        self.current_maze = maze
        policy = self.get_policy()
        
        # Visualization code here (same as in original)
        import matplotlib.pyplot as plt
        
        directions = ['↑', '↓', '←', '→']
        
        plt.figure(figsize=(10, 10))
        plt.imshow(maze, cmap='binary')
        
        for r in range(self.state_size[0]):
            for c in range(self.state_size[1]):
                if maze[r, c] == 0:
                    plt.text(c, r, directions[policy[r, c]], 
                           ha='center', va='center', color='blue', fontsize=12)
        
        plt.title('DQN Policy')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def visualize_value_function(self, maze: np.ndarray):
        """
        Visualize the value function.
        """
        self.current_maze = maze
        values = self.get_value_function()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(maze, cmap='binary')
        
        for r in range(self.state_size[0]):
            for c in range(self.state_size[1]):
                if maze[r, c] == 0:
                    color = 'green' if values[r, c] > 0 else 'red'
                    plt.text(c, r, f"{values[r, c]:.1f}", 
                           ha='center', va='center', color=color, fontsize=8)
        
        plt.title('DQN Value Function')
        plt.grid(True)
        plt.tight_layout()
        plt.show()