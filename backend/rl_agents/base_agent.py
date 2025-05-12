# base_agent.py

from abc import ABC, abstractmethod
import numpy as np 
import os 
import pickle
from typing import Tuple, Dict, List, Any, Optional



class BaseAgent(ABC):
    """
    Lớp cơ sở trừu tượng cho các thuật toán học tăng cường.
    
    Các thuật toán như Q-Learning và SARSA sẽ kế thừa từ lớp này
    và cài đặt các phương thức cần thiết cho việc học tăng cường.
    """
    
    def __init__(self, state_size: Tuple[int, int], action_size: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 exploration_rate: float = 1.0, exploration_decay = 0.995,
                 min_exploration_rate: float = 0.01, seed: Optional[int] = None):
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
        Giảm tỷ lệ khám phá (epsilon) theo thời gian.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
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
            'steps_per_episode': self.steps_per_episode
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
        
    def export_to_json(self, file_path: str) -> None: 
        """
        Xuất mô hình sang định dạng JSON để sử dụng trên web.
        
        Args:
            file_path (str): Đường dẫn đến file JSON
        """
        import json
        
        # Chuyển đổi Q-table thành định dạng dễ hiểu trong JavaScript
        q_table_list = []
        height, width, actions = self.q_table.shape
        
        for r in range(height):
            for c in range(width):
                q_values = self.q_table[r, c, :].tolist()
                q_table_list.append({
                    'row': r,
                    'col': c,
                    'q_values': q_values,
                    'best_action': int(np.argmax(q_values))
                })
        
        # Tạo đối tượng JSON
        model_json = {
            'state_size': list(self.state_size),
            'action_size': self.action_size,
            'q_table': q_table_list
        }
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Lưu vào file JSON
        with open(file_path, 'w') as f:
            json.dump(model_json, f, indent=2)
            
    def plot_training_progress(self) -> None:
        """
        Vẽ biểu đồ tiến trình huấn luyện.
        """
        import matplotlib.pyplot as plt
        
        # Vẽ biểu đồ phần thưởng theo episode
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Phần thưởng theo episode')
        plt.xlabel('Episode')
        plt.ylabel('Tổng phần thưởng')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.steps_per_episode)
        plt.title('Số bước theo episode')
        plt.xlabel('Episode')
        plt.ylabel('Số bước')
        
        plt.tight_layout()
        plt.show()