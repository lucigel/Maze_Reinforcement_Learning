# base_agent.py

from abc import ABC, abstractmethod
import numpy as np 
import os 
import pickle
from typing import Tuple, Dict, List, Any, Optional
import random  # Thêm import

class BaseAgent(ABC):
    """
    Lớp cơ sở trừu tượng cho các thuật toán học tăng cường.
    
    Các thuật toán như Q-Learning và SARSA sẽ kế thừa từ lớp này
    và cài đặt các phương thức cần thiết cho việc học tăng cường.
    """
    
    def __init__(self, state_size: Tuple[int, int], action_size: int, 
                 learning_rate: float = 0.2, discount_factor: float = 0.99, 
                 exploration_rate: float = 1.0, exploration_decay = 0.998,
                 min_exploration_rate: float = 0.05, seed: Optional[int] = None,
                 buffer_size: int = 10000):
        """
        Khởi tạo agent học tăng cường cơ bản.
        
        Args:
            state_size (Tuple[int, int]): Kích thước không gian trạng thái (height, width)
            action_size (int): Số lượng hành động có thể thực hiện
            learning_rate (float): Tốc độ học (alpha)
            discount_factor (float): Hệ số giảm (gamma)
            exploration_rate (float): Tỷ lệ khám phá ban đầu (epsilon)
            exploration_decay (float): Tốc độ giảm tỷ lệ khám phá
            min_exploration_rate (float): Giá trị nhỏ nhất của tỷ lệ khám phá
            seed (int, optional): Hạt giống cho bộ sinh số ngẫu nhiên
            buffer_size (int): Kích thước buffer cho experience replay
            
            Q(s, a) = Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate
        
        self.rng = np.random.RandomState(seed)
        
        height, width = state_size
        self.q_table = np.zeros((height, width, action_size))
        
        self.episode_rewards = []
        self.steps_per_episode = []
        
        # Thêm các biến cho experience replay
        self.experience_buffer = []
        self.buffer_size = buffer_size
        
        # Thêm biến theo dõi cho state visitation
        self.state_visits = np.zeros((height, width))
    
    @abstractmethod
    def choose_action(self, state: Tuple[int, int]) -> int: 
        """
        Chọn hành động dựa trên trạng thái hiện tại.
        
        Args:
            state (Tuple[int, int]): Trạng thái hiện tại (row, col)
            
        Returns:
            int: Hành động được chọn
        """
        pass 
    
    
    @abstractmethod
    def learn(self, state: Tuple[int, int], action: int, 
              reward: float, next_state: Tuple[int, int], 
              done: bool, next_action: Optional[int] = None) -> None: 
        
        """
        Cập nhật kiến thức của agent dựa trên trải nghiệm.
        
        Args:
            state (Tuple[int, int]): Trạng thái hiện tại
            action (int): Hành động được thực hiện
            reward (float): Phần thưởng nhận được
            next_state (Tuple[int, int]): Trạng thái tiếp theo
            done (bool): True nếu episode kết thúc
            next_action (int, optional): Hành động tiếp theo (chỉ dùng cho SARSA)
        """
        pass
    
    def decay_exploration(self) -> None: 
        """
        Giảm tỷ lệ khám phá (epsilon) theo thời gian với cơ chế thích ứng.
        """
        # Giảm epsilon chậm hơn ở giai đoạn đầu, nhanh hơn về sau
        if self.epsilon > 0.5:
            self.epsilon = max(self.min_epsilon, self.epsilon * 0.999)  # Giảm chậm hơn
        else:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def add_experience(self, experience: Tuple) -> None:
        """
        Thêm một trải nghiệm vào buffer.
        
        Args:
            experience (Tuple): Tuple (state, action, reward, next_state, done)
        """
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
    
    def replay_experiences(self, batch_size: int = 32) -> None:
        """
        Học lại từ các trải nghiệm đã lưu trữ.
        
        Args:
            batch_size (int): Số lượng trải nghiệm học lại mỗi lần
        """
        if len(self.experience_buffer) < batch_size:
            return
            
        experiences = random.sample(self.experience_buffer, batch_size)
        for state, action, reward, next_state, done in experiences:
            self.learn(state, action, reward, next_state, done)
            
    def save_model(self, file_path: str) -> None:
        """
        Lưu mô hình (Q-table) vào file.
        
        Args:
            file_path (str): Đường dẫn đến file
        """
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        model_data = {
            'q_table': self.q_table, 
            'state_size': self.state_size, 
            'action_size': self.action_size, 
            'epsilon': self.epsilon, 
            'lr': self.lr, 
            'gamma': self.gamma, 
            'episode_rewards': self.episode_rewards, 
            'steps_per_episode': self.steps_per_episode,
            'state_visits': self.state_visits
        }
        
        with open(file_path, 'wb') as f: 
            pickle.dump(model_data, f)
            
    def load_model(self, file_path: str) -> None: 
        """
        Tải mô hình từ file.
        
        Args:
            file_path (str): Đường dẫn đến file
        """
        with open(file_path, 'rb') as f: 
            model_data = pickle.load(f)
            
        self.q_table = model_data['q_table']
        self.state_size = model_data['state_size']
        self.action_size = model_data['action_size']
        self.epsilon = model_data['epsilon']
        self.lr = model_data['lr']
        self.gamma = model_data['gamma']
        self.episode_rewards = model_data['episode_rewards']
        self.steps_per_episode = model_data['steps_per_episode']
        
        # Tải state_visits nếu có
        if 'state_visits' in model_data:
            self.state_visits = model_data['state_visits']
        else:
            # Nếu đang tải mô hình cũ
            self.state_visits = np.zeros(self.state_size)