import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from backend.rl_agents.base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    """
    Triển khai thuật toán Q-Learning cho bài toán mê cung.
    
    Q-Learning là thuật toán học tăng cường off-policy, học từ những trải nghiệm
    không trực tiếp liên quan đến chính sách hiện tại của agent. Thuật toán này
    sử dụng công thức cập nhật:
    
    Q(s, a) = Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]
    
    Trong đó:
    - s, a: Trạng thái và hành động hiện tại
    - s': Trạng thái tiếp theo sau khi thực hiện hành động a
    - r: Phần thưởng nhận được
    - α: Tốc độ học (learning rate)
    - γ: Hệ số giảm (discount factor)
    - max Q(s', a'): Giá trị Q lớn nhất có thể từ trạng thái s'
    """
    
    def __init__(self, state_size: Tuple[int, int], action_size: int = 4, 
                 learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995,
                 min_exploration_rate: float = 0.01, seed: Optional[int] = None):
        """
        Khởi tạo agent Q-Learning.
        
        Args:
            state_size (Tuple[int, int]): Kích thước không gian trạng thái (height, width)
            action_size (int): Số lượng hành động có thể thực hiện
            learning_rate (float): Tốc độ học (alpha)
            discount_factor (float): Hệ số giảm (gamma)
            exploration_rate (float): Tỷ lệ khám phá ban đầu (epsilon)
            exploration_decay (float): Tốc độ giảm tỷ lệ khám phá
            min_exploration_rate (float): Giá trị nhỏ nhất của tỷ lệ khám phá
            seed (int, optional): Hạt giống cho bộ sinh số ngẫu nhiên
        """
        super().__init__(state_size, action_size, learning_rate, discount_factor,
                       exploration_rate, exploration_decay, min_exploration_rate, seed)
        
        # Các bảng thống kê bổ sung
        self.state_visits = np.zeros(state_size)  # Theo dõi số lần thăm mỗi trạng thái
        self.action_counts = np.zeros((state_size[0], state_size[1], action_size))  # Số lần thực hiện mỗi hành động
        
    def choose_action(self, state: Tuple[int, int]) -> int:
        """
        Chọn hành động sử dụng chiến lược epsilon-greedy.
        
        Args:
            state (Tuple[int, int]): Trạng thái hiện tại (row, col)
            
        Returns:
            int: Hành động được chọn
        """
        # Tăng số lần thăm trạng thái này
        self.state_visits[state] += 1
        
        # Khám phá ngẫu nhiên với xác suất epsilon
        if self.rng.random() < self.epsilon:
            action = self.rng.randint(0, self.action_size)
        else:
            # Khai thác (chọn hành động có giá trị Q cao nhất)
            action = np.argmax(self.q_table[state])
        
        # Tăng số lần thực hiện hành động này
        self.action_counts[state[0], state[1], action] += 1
        
        return action
    
    def learn(self, state: Tuple[int, int], action: int, 
              reward: float, next_state: Tuple[int, int], 
              done: bool, next_action: Optional[int] = None) -> None:
        """
        Cập nhật Q-table sử dụng thuật toán Q-Learning.
        
        Args:
            state (Tuple[int, int]): Trạng thái hiện tại
            action (int): Hành động được thực hiện
            reward (float): Phần thưởng nhận được
            next_state (Tuple[int, int]): Trạng thái tiếp theo
            done (bool): True nếu episode kết thúc
            next_action (int, optional): Không sử dụng trong Q-Learning, chỉ để tương thích với BaseRLAgent
        """
        # Lấy giá trị Q hiện tại cho cặp (state, action)
        current_q = self.q_table[state[0], state[1], action]
        
        # Tính toán giá trị Q mới
        if done:
            # Nếu đã kết thúc episode, không có trạng thái tiếp theo
            max_next_q = 0
        else:
            # Chọn hành động tốt nhất từ trạng thái tiếp theo (không phụ thuộc vào chính sách)
            max_next_q = np.max(self.q_table[next_state[0], next_state[1]])
        
        # Tính toán mục tiêu Q dựa trên công thức Q-Learning
        target_q = reward + self.gamma * max_next_q
        
        # Cập nhật giá trị Q
        self.q_table[state[0], state[1], action] += self.lr * (target_q - current_q)
    
    def train(self, env, num_episodes: int = 1000, max_steps: int = 1000,
              verbose: bool = True, save_path: Optional[str] = None,
              save_interval: int = 100) -> Dict[str, List]:
        """
        Huấn luyện agent sử dụng thuật toán Q-Learning.
        
        Args:
            env: Môi trường mê cung
            num_episodes (int): Số lượng episode huấn luyện
            max_steps (int): Số bước tối đa trong mỗi episode
            verbose (bool): Hiển thị thông tin huấn luyện
            save_path (str, optional): Đường dẫn lưu mô hình
            save_interval (int): Khoảng thời gian lưu mô hình
            
        Returns:
            Dict[str, List]: Kết quả huấn luyện (reward, steps)
        """
        # Khởi tạo lại danh sách theo dõi
        self.episode_rewards = []
        self.steps_per_episode = []
        
        # Huấn luyện qua các episode
        for episode in range(num_episodes):
            # Reset môi trường
            state = env.reset()
            episode_reward = 0
            step = 0
            
            while step < max_steps:
                # Chọn hành động
                action = self.choose_action(state)
                
                # Thực hiện hành động
                next_state, reward, done, _ = env.step(action)
                
                # Học từ trải nghiệm
                self.learn(state, action, reward, next_state, done)
                
                # Cập nhật trạng thái
                state = next_state
                episode_reward += reward
                step += 1
                
                # Nếu đã đạt đến đích, kết thúc episode
                if done:
                    break
            
            # Lưu thông tin episode
            self.episode_rewards.append(episode_reward)
            self.steps_per_episode.append(step)
            
            # Giảm tỷ lệ khám phá
            self.decay_exploration()
            
            # Hiển thị tiến độ
            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {step}, Epsilon: {self.epsilon:.4f}")
            
            # Lưu mô hình định kỳ
            if save_path and (episode + 1) % save_interval == 0:
                self.save_model(f"{save_path}/q_learning_episode_{episode + 1}.pkl")
        
        # Lưu mô hình cuối cùng
        if save_path:
            self.save_model(f"{save_path}/q_learning_final.pkl")
        
        # Trả về kết quả huấn luyện
        return {
            "rewards": self.episode_rewards,
            "steps": self.steps_per_episode
        }
    
    def get_policy(self) -> np.ndarray:
        """
        Lấy chính sách tốt nhất từ Q-table.
        
        Returns:
            np.ndarray: Ma trận chứa hành động tốt nhất cho mỗi trạng thái
        """
        height, width = self.state_size
        policy = np.zeros((height, width), dtype=int)
        
        for r in range(height):
            for c in range(width):
                policy[r, c] = np.argmax(self.q_table[r, c])
        
        return policy
    
    def get_value_function(self) -> np.ndarray:
        """
        Lấy hàm giá trị từ Q-table.
        
        Returns:
            np.ndarray: Ma trận chứa giá trị tốt nhất cho mỗi trạng thái
        """
        height, width = self.state_size
        value_function = np.zeros((height, width))
        
        for r in range(height):
            for c in range(width):
                value_function[r, c] = np.max(self.q_table[r, c])
        
        return value_function
    
    def visualize_policy(self, maze: np.ndarray) -> None:
        """
        Hiển thị chính sách tốt nhất trên mê cung.
        
        Args:
            maze (np.ndarray): Ma trận mê cung
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        # Lấy chính sách tốt nhất
        policy = self.get_policy()
        
        # Tạo bản sao mê cung
        maze_copy = maze.copy()
        
        # Danh sách các ký hiệu hướng
        directions = ['↑', '↓', '←', '→']
        
        # Kích thước mê cung
        height, width = maze.shape
        
        plt.figure(figsize=(10, 10))
        
        # Hiển thị mê cung
        cmap = ListedColormap(['white', 'black', 'green', 'red'])
        plt.imshow(maze_copy, cmap=cmap)
        
        # Hiển thị chính sách
        for r in range(height):
            for c in range(width):
                if maze_copy[r, c] == 0:  # Chỉ hiển thị chính sách ở các ô đường đi
                    plt.text(c, r, directions[policy[r, c]], 
                             ha='center', va='center', color='blue', fontsize=12)
        
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.title('Chính sách tốt nhất')
        plt.tight_layout()
        plt.show()
    
    def visualize_value_function(self, maze: np.ndarray) -> None:
        """
        Hiển thị hàm giá trị trên mê cung.
        
        Args:
            maze (np.ndarray): Ma trận mê cung
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        
        # Lấy hàm giá trị
        value_function = self.get_value_function()
        
        # Tạo bản sao mê cung
        maze_copy = maze.copy()
        
        # Kích thước mê cung
        height, width = maze.shape
        
        plt.figure(figsize=(10, 10))
        
        # Hiển thị mê cung
        plt.imshow(maze_copy, cmap='binary')
        
        # Hiển thị hàm giá trị
        for r in range(height):
            for c in range(width):
                if maze_copy[r, c] == 0:  # Chỉ hiển thị giá trị ở các ô đường đi
                    color = 'green' if value_function[r, c] > 0 else 'red'
                    plt.text(c, r, f"{value_function[r, c]:.1f}", 
                             ha='center', va='center', color=color, fontsize=8)
        
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.title('Hàm giá trị')
        plt.tight_layout()
        plt.show()