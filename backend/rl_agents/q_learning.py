# q_learning.py
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from rl_agents.base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    """
    Triển khai thuật toán Q-Learning cho bài toán mê cung với các cải tiến.
    
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
                 learning_rate: float = 0.2, discount_factor: float = 0.99, 
                 exploration_rate: float = 1.0, exploration_decay: float = 0.998,
                 min_exploration_rate: float = 0.05, seed: Optional[int] = None,
                 use_double_q: bool = True):
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
            use_double_q (bool): Sử dụng Double Q-Learning
        """
        super().__init__(state_size, action_size, learning_rate, discount_factor,
                       exploration_rate, exploration_decay, min_exploration_rate, seed)
        
        # Các bảng thống kê bổ sung
        self.state_visits = np.zeros(state_size)  # Theo dõi số lần thăm mỗi trạng thái
        self.action_counts = np.zeros((state_size[0], state_size[1], action_size))  # Số lần thực hiện mỗi hành động
        
        # Tham số cho Double Q-Learning
        self.use_double_q = use_double_q
        if use_double_q:
            self.target_q_table = np.zeros((state_size[0], state_size[1], action_size))
            self.update_target_steps = 500  # Cập nhật target network mỗi 500 steps
            self.steps = 0
    
    def choose_action(self, state: Tuple[int, int]) -> int:
        """
        Chọn hành động với chiến lược epsilon-greedy được cải tiến.
        
        Args:
            state (Tuple[int, int]): Trạng thái hiện tại (row, col)
            
        Returns:
            int: Hành động được chọn
        """
        # Tăng số lần thăm trạng thái này
        self.state_visits[state] += 1
        
        # Tìm các hành động dẫn đến trạng thái chưa thăm
        row, col = state
        unvisited_actions = []
        
        for action in range(self.action_size):
            # Các hướng: 0: lên, 1: xuống, 2: trái, 3: phải
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dr, dc = directions[action]
            new_row, new_col = row + dr, col + dc
            
            # Kiểm tra nếu vị trí mới hợp lệ và chưa được thăm nhiều
            if 0 <= new_row < self.state_size[0] and 0 <= new_col < self.state_size[1]:
                if self.state_visits[new_row, new_col] < 5:  # Ưu tiên ô chưa thăm hoặc thăm ít
                    unvisited_actions.append(action)
        
        # Ưu tiên khám phá các trạng thái chưa thăm
        if unvisited_actions and self.rng.random() < self.epsilon * 1.5:  # Tăng xác suất khám phá
            return self.rng.choice(unvisited_actions)
        # Khám phá ngẫu nhiên với xác suất epsilon
        elif self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.action_size)
        else:
            # Khai thác (chọn hành động có giá trị Q cao nhất)
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            
            # Nếu có nhiều hành động có giá trị Q bằng nhau, chọn ngẫu nhiên một trong số đó
            actions_with_max_q = np.where(q_values == max_q)[0]
            action = self.rng.choice(actions_with_max_q)
        
        # Tăng số lần thực hiện hành động này
        self.action_counts[state[0], state[1], action] += 1
        
        return action
    
    def learn(self, state: Tuple[int, int], action: int, 
              reward: float, next_state: Tuple[int, int], 
              done: bool, next_action: Optional[int] = None) -> None:
        """
        Cập nhật Q-table sử dụng thuật toán Q-Learning cải tiến.
        
        Args:
            state (Tuple[int, int]): Trạng thái hiện tại
            action (int): Hành động được thực hiện
            reward (float): Phần thưởng nhận được
            next_state (Tuple[int, int]): Trạng thái tiếp theo
            done (bool): True nếu episode kết thúc
            next_action (int, optional): Không sử dụng trong Q-Learning
        """
        # Lấy giá trị Q hiện tại cho cặp (state, action)
        current_q = self.q_table[state[0], state[1], action]
        
        # Tính toán giá trị Q mới
        if done:
            # Nếu đã kết thúc episode, không có trạng thái tiếp theo
            max_next_q = 0
        else:
            if self.use_double_q:
                # Double Q-Learning: Chọn action từ Q chính, nhưng lấy giá trị từ Q target
                best_action = np.argmax(self.q_table[next_state[0], next_state[1]])
                max_next_q = self.target_q_table[next_state[0], next_state[1], best_action]
            else:
                # Q-Learning tiêu chuẩn
                max_next_q = np.max(self.q_table[next_state[0], next_state[1]])
        
        # Tính toán mục tiêu Q dựa trên công thức Q-Learning
        target_q = reward + self.gamma * max_next_q
        
        # Cập nhật giá trị Q với tốc độ học thích ứng
        # Giảm learning rate cho các trạng thái đã thăm nhiều lần
        adaptive_lr = self.lr / (1 + 0.1 * self.state_visits[state])
        self.q_table[state[0], state[1], action] += adaptive_lr * (target_q - current_q)
        
        # Cập nhật target network nếu sử dụng Double Q-Learning
        if self.use_double_q:
            self.steps += 1
            if self.steps % self.update_target_steps == 0:
                self.target_q_table = self.q_table.copy()
        
        # Lưu trải nghiệm vào buffer
        self.add_experience((state, action, reward, next_state, done))
    
    def train(self, env, num_episodes: int = 1000, max_steps: int = 1000,
              verbose: bool = True, save_path: Optional[str] = None,
              save_interval: int = 100, replay_batch_size: int = 32) -> Dict[str, List]:
        """
        Huấn luyện agent sử dụng thuật toán Q-Learning.
        
        Args:
            env: Môi trường mê cung
            num_episodes (int): Số lượng episode huấn luyện
            max_steps (int): Số bước tối đa trong mỗi episode
            verbose (bool): Hiển thị thông tin huấn luyện
            save_path (str, optional): Đường dẫn lưu mô hình
            save_interval (int): Khoảng thời gian lưu mô hình
            replay_batch_size (int): Kích thước batch cho experience replay
            
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
                next_state, reward, done, info = env.step(action)
                
                # Học từ trải nghiệm
                self.learn(state, action, reward, next_state, done)
                
                # Experience replay mỗi 10 bước
                if step % 10 == 0 and len(self.experience_buffer) > replay_batch_size:
                    self.replay_experiences(batch_size=replay_batch_size)
                
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