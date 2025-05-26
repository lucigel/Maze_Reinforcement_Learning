# sarsa.py
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from rl_agents.base_agent import BaseAgent

class SARSAAgent(BaseAgent):
    """
    Triển khai thuật toán SARSA cho bài toán mê cung với các cải tiến.
    
    SARSA (State-Action-Reward-State-Action) là thuật toán học tăng cường on-policy,
    học từ những trải nghiệm trực tiếp liên quan đến chính sách hiện tại của agent.
    """
    
    def __init__(self, state_size: Tuple[int, int], action_size: int = 4, 
                 learning_rate: float = 0.2, discount_factor: float = 0.99, 
                 exploration_rate: float = 1.0, exploration_decay: float = 0.98,
                 min_exploration_rate: float = 0.05, seed: Optional[int] = None,
                 use_expected_sarsa: bool = True):
        """
        Khởi tạo agent SARSA với các cải tiến.
        
        Args:
            state_size (Tuple[int, int]): Kích thước không gian trạng thái (height, width)
            action_size (int): Số lượng hành động có thể thực hiện
            learning_rate (float): Tốc độ học (alpha)
            discount_factor (float): Hệ số giảm (gamma)
            exploration_rate (float): Tỷ lệ khám phá ban đầu (epsilon)
            exploration_decay (float): Tốc độ giảm tỷ lệ khám phá
            min_exploration_rate (float): Giá trị nhỏ nhất của tỷ lệ khám phá
            seed (int, optional): Hạt giống cho bộ sinh số ngẫu nhiên
            use_expected_sarsa (bool): Sử dụng Expected SARSA thay vì SARSA thông thường
        """
        super().__init__(state_size, action_size, learning_rate, discount_factor,
                       exploration_rate, exploration_decay, min_exploration_rate, seed)
        
        # Khởi tạo Q-table với giá trị lạc quan (optimistic) giúp khuyến khích khám phá ban đầu
        height, width = state_size
        self.q_table = np.ones((height, width, action_size)) * 0.1
        
        # Các bảng thống kê bổ sung
        self.state_visits = np.zeros(state_size)  # Theo dõi số lần thăm mỗi trạng thái
        self.action_counts = np.zeros((state_size[0], state_size[1], action_size))  # Số lần thực hiện mỗi hành động
        self.state_action_counts = np.zeros((state_size[0], state_size[1], action_size))  # Số lần thực hiện cặp (s,a)
        
        # Tham số cho Expected SARSA
        self.use_expected_sarsa = use_expected_sarsa
        
        # Lưu trữ các trạng thái gần đây để phát hiện vòng lặp
        self.recent_states = []  # Lưu các trạng thái gần đây
        self.max_recent_states = 10  # Số lượng trạng thái gần đây tối đa để lưu
        
        # Biến theo dõi cho thống kê huấn luyện
        self.episode_success = []  # Theo dõi episode thành công (đạt đích)
        self.td_errors = []  # Lưu TD errors cho prioritized replay
        
        # Cài đặt Exploration/Exploitation
        self.ucb_c = 0.5  # Hệ số cho Upper Confidence Bound
        self.use_intrinsic_reward = True  # Sử dụng phần thưởng nội tại để khuyến khích khám phá
        self.intrinsic_reward_scale = 0.2  # Hệ số cho phần thưởng nội tại
    
    def choose_action(self, state: Tuple[int, int]) -> int:
        """
        Chọn hành động với chiến lược khám phá được cải tiến.
        
        Args:
            state (Tuple[int, int]): Trạng thái hiện tại (row, col)
            
        Returns:
            int: Hành động được chọn
        """
        # Tăng số lần thăm trạng thái này
        self.state_visits[state] += 1
        row, col = state
        
        # Chiến lược 1: Khai thác với xác suất (1 - epsilon)
        if self.rng.random() > self.epsilon:
            # Khai thác: chọn hành động tốt nhất dựa trên Q-table
            q_values = self.q_table[row, col]
            max_q = np.max(q_values)
            actions_with_max_q = np.where(q_values == max_q)[0]
            
            # Nếu có nhiều hành động tốt nhất, chọn ngẫu nhiên một trong số đó
            return self.rng.choice(actions_with_max_q)
        
        # Chiến lược 2: Ưu tiên khám phá các trạng thái mới/ít thăm
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Lên, xuống, trái, phải
        valid_actions = []
        unvisited_actions = []
        
        for action in range(self.action_size):
            dr, dc = directions[action]
            new_row, new_col = row + dr, col + dc
            
            # Kiểm tra xem vị trí mới có hợp lệ không
            if 0 <= new_row < self.state_size[0] and 0 <= new_col < self.state_size[1]:
                valid_actions.append(action)
                
                # Nếu trạng thái mới ít được thăm, thêm vào danh sách ưu tiên
                if self.state_visits[new_row, new_col] < 3:
                    unvisited_actions.append(action)
        
        # Nếu có các trạng thái ít thăm và random_value < 0.7, ưu tiên chọn chúng
        if unvisited_actions and self.rng.random() < 0.7:
            return self.rng.choice(unvisited_actions)
        
        # Chiến lược 3: Khám phá hướng đến đích (nếu biết vị trí đích)
        if hasattr(self, 'goal_pos') and self.rng.random() < 0.4:
            goal_row, goal_col = self.goal_pos
            actions_toward_goal = []
            
            for action in valid_actions:
                dr, dc = directions[action]
                new_row, new_col = row + dr, col + dc
                
                # Tính khoảng cách Manhattan đến đích
                current_dist = abs(row - goal_row) + abs(col - goal_col)
                new_dist = abs(new_row - goal_row) + abs(new_col - goal_col)
                
                # Nếu hành động giảm khoảng cách đến đích, thêm vào danh sách
                if new_dist < current_dist:
                    actions_toward_goal.append(action)
            
            if actions_toward_goal:
                return self.rng.choice(actions_toward_goal)
        
        # Chiến lược 4: Tránh quay lại trạng thái gần đây (tránh lặp)
        if self.recent_states and self.rng.random() < 0.3:
            avoid_actions = []
            
            for action in valid_actions:
                dr, dc = directions[action]
                new_row, new_col = row + dr, col + dc
                new_state = (new_row, new_col)
                
                # Nếu hành động dẫn đến trạng thái không nằm trong list gần đây, thêm vào
                if new_state not in self.recent_states[-3:]:  # Chỉ xét 3 trạng thái gần nhất
                    avoid_actions.append(action)
            
            if avoid_actions:
                return self.rng.choice(avoid_actions)
        
        # Chiến lược 5: Khám phá ngẫu nhiên đơn thuần
        if valid_actions:
            return self.rng.choice(valid_actions)
        else:
            return self.rng.randint(0, self.action_size)
    
    def learn(self, state: Tuple[int, int], action: int, 
              reward: float, next_state: Tuple[int, int], 
              done: bool, next_action: Optional[int] = None) -> None:
        """
        Cập nhật Q-table sử dụng thuật toán SARSA với các cải tiến.
        
        Args:
            state (Tuple[int, int]): Trạng thái hiện tại
            action (int): Hành động được thực hiện
            reward (float): Phần thưởng nhận được
            next_state (Tuple[int, int]): Trạng thái tiếp theo
            done (bool): True nếu episode kết thúc
            next_action (int, optional): Hành động tiếp theo
        """
        # Cập nhật danh sách trạng thái gần đây
        self.recent_states.append(state)
        if len(self.recent_states) > self.max_recent_states:
            self.recent_states.pop(0)  # Loại bỏ trạng thái cũ nhất
        
        # Tính phần thưởng tổng (thêm phần thưởng nội tại)
        total_reward = reward
        
        # Thêm phần thưởng nội tại nếu được kích hoạt
        if self.use_intrinsic_reward:
            # Phần thưởng cho sự mới lạ (novelty)
            visit_count = self.state_visits[next_state]
            novelty_reward = 1.0 / (visit_count + 1)
            
            # Phạt nếu quay lại trạng thái gần đây (tránh lặp)
            loop_penalty = 0
            if next_state in self.recent_states[-5:]:  # Kiểm tra 5 trạng thái gần nhất
                loop_penalty = -0.2
            
            # Thêm vào phần thưởng tổng
            total_reward += self.intrinsic_reward_scale * (novelty_reward + loop_penalty)
        
        # Tăng số lần thực hiện hành động này từ trạng thái này
        self.action_counts[state[0], state[1], action] += 1
        self.state_action_counts[state[0], state[1], action] += 1
        
        # Lấy giá trị Q hiện tại cho cặp (state, action)
        current_q = self.q_table[state[0], state[1], action]
        
        # Tính toán TD target
        if done:
            # Nếu đã kết thúc episode, không có giá trị tiếp theo
            td_target = total_reward
        else:
            if self.use_expected_sarsa:
                # Expected SARSA: tính giá trị kỳ vọng của Q(s', a')
                td_target = self._calculate_expected_sarsa_target(total_reward, next_state)
            else:
                # SARSA thông thường
                if next_action is None:
                    next_action = self.choose_action(next_state)
                
                td_target = total_reward + self.gamma * self.q_table[next_state[0], next_state[1], next_action]
        
        # Tính TD error
        td_error = td_target - current_q
        
        # Tốc độ học thích ứng (giảm khi trạng thái được thăm nhiều lần)
        adaptive_lr = max(0.05, self.lr / (1 + 0.05 * self.state_visits[state]))
        
        # Cập nhật giá trị Q
        self.q_table[state[0], state[1], action] += adaptive_lr * td_error
        
        # Lưu TD error cho prioritized replay
        self.td_errors.append((state, action, reward, next_state, done, abs(td_error)))
        if len(self.td_errors) > 10000:
            self.td_errors = self.td_errors[-10000:]  # Giữ độ dài buffer ở mức hợp lý
        
        # Lưu trải nghiệm vào buffer
        self.add_experience((state, action, reward, next_state, done))
    
    def _calculate_expected_sarsa_target(self, reward: float, next_state: Tuple[int, int]) -> float:
        """
        Tính TD target cho Expected SARSA.
        
        Args:
            reward (float): Phần thưởng nhận được
            next_state (Tuple[int, int]): Trạng thái tiếp theo
            
        Returns:
            float: Expected SARSA target
        """
        q_values = self.q_table[next_state[0], next_state[1]]
        max_action = np.argmax(q_values)
        
        # Tính xác suất chọn mỗi hành động theo chính sách epsilon-greedy
        greedy_prob = 1 - self.epsilon + self.epsilon / self.action_size
        non_greedy_prob = self.epsilon / self.action_size
        
        # Tính giá trị kỳ vọng
        expected_q = 0
        for a in range(self.action_size):
            if a == max_action:
                expected_q += greedy_prob * q_values[a]
            else:
                expected_q += non_greedy_prob * q_values[a]
        
        return reward + self.gamma * expected_q
    
    def prioritized_replay(self, batch_size: int = 32) -> None:
        """
        Học lại từ các trải nghiệm với ưu tiên dựa trên lỗi TD.
        
        Args:
            batch_size (int): Kích thước batch
        """
        if len(self.td_errors) < batch_size:
            return
        
        # Tính xác suất lấy mẫu dựa trên TD error
        errors = np.array([err for _, _, _, _, _, err in self.td_errors])
        probs = errors / np.sum(errors)
        
        # Lấy mẫu dựa trên xác suất
        indices = self.rng.choice(len(self.td_errors), size=batch_size, p=probs)
        
        # Học từ các mẫu được chọn
        for idx in indices:
            state, action, reward, next_state, done, _ = self.td_errors[idx]
            self.learn(state, action, reward, next_state, done)
    
    def adaptive_exploration_decay(self) -> None:
        """
        Điều chỉnh tỷ lệ khám phá dựa trên hiệu suất gần đây.
        """
        # Đánh giá hiệu suất dựa trên episode gần đây
        window_size = 10  # Số episode gần nhất để xem xét
        
        if len(self.episode_success) >= window_size:
            # Tính tỷ lệ thành công gần đây
            recent_success = np.mean(self.episode_success[-window_size:])
            
            # Điều chỉnh epsilon dựa trên hiệu suất
            if recent_success > 0.7:
                # Hiệu suất tốt, giảm epsilon nhanh hơn
                self.epsilon = max(self.min_epsilon, self.epsilon * 0.9)
            elif recent_success > 0.3:
                # Hiệu suất trung bình, giảm epsilon bình thường
                self.epsilon = max(self.min_epsilon, self.epsilon * 0.95)
            else:
                # Hiệu suất kém, giảm epsilon chậm hơn
                self.epsilon = max(self.min_epsilon, self.epsilon * 0.98)
        else:
            # Chưa đủ dữ liệu, sử dụng decay mặc định
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def train(self, env, num_episodes: int = 1000, max_steps: int = 1000,
              verbose: bool = True, save_path: Optional[str] = None,
              save_interval: int = 100, replay_batch_size: int = 32) -> Dict[str, List]:
        """
        Huấn luyện agent sử dụng thuật toán SARSA cải tiến.
        
        Args:
            env: Môi trường mê cung
            num_episodes (int): Số lượng episode huấn luyện
            max_steps (int): Số bước tối đa trong mỗi episode
            verbose (bool): Hiển thị thông tin huấn luyện
            save_path (str, optional): Đường dẫn lưu mô hình
            save_interval (int): Khoảng thời gian lưu mô hình
            replay_batch_size (int): Kích thước batch cho experience replay
            
        Returns:
            Dict[str, List]: Kết quả huấn luyện (reward, steps, success_rate)
        """
        # Lưu thông tin môi trường (sử dụng cho heuristic)
        if hasattr(env, 'goal_pos'):
            self.goal_pos = env.goal_pos
        
        # Khởi tạo lại danh sách theo dõi
        self.episode_rewards = []
        self.steps_per_episode = []
        self.episode_success = []
        
        # Huấn luyện qua các episode
        for episode in range(num_episodes):
            # Reset môi trường và biến theo dõi
            state = env.reset()
            self.recent_states = []  # Reset lịch sử trạng thái
            episode_reward = 0
            step = 0
            
            # Chọn hành động đầu tiên
            action = self.choose_action(state)
            
            # Đếm số trạng thái đã thăm trong episode này
            visited_states = set()
            stuck_counter = 0  # Đếm số bước bị kẹt
            
            while step < max_steps:
                # Thực hiện hành động
                next_state, reward, done, _ = env.step(action)
                
                # Theo dõi trạng thái đã thăm để phát hiện bị kẹt
                state_key = (next_state[0], next_state[1])
                if state_key in visited_states:
                    stuck_counter += 1
                else:
                    visited_states.add(state_key)
                    stuck_counter = 0
                
                # Nếu agent bị kẹt quá lâu, tăng epsilon tạm thời để thúc đẩy khám phá
                if stuck_counter > 50:
                    self.epsilon = min(1.0, self.epsilon * 1.5)
                    stuck_counter = 0
                
                # Chọn hành động tiếp theo (cần thiết cho SARSA)
                next_action = self.choose_action(next_state) if not done else None
                
                # Học từ trải nghiệm
                self.learn(state, action, reward, next_state, done, next_action)
                
                # Experience replay mỗi 5 bước
                if step % 5 == 0 and len(self.td_errors) > replay_batch_size:
                    self.prioritized_replay(batch_size=replay_batch_size)
                
                # Cập nhật trạng thái và hành động
                state = next_state
                action = next_action if next_action is not None else 0
                episode_reward += reward
                step += 1
                
                # Nếu đã đạt đến đích, kết thúc episode
                if done:
                    # Đánh dấu episode thành công nếu reward dương (đạt đích)
                    self.episode_success.append(1 if reward > 0 else 0)
                    break
            
            # Nếu không kết thúc sau max_steps, đánh dấu là thất bại
            if not done:
                self.episode_success.append(0)
            
            # Lưu thông tin episode
            self.episode_rewards.append(episode_reward)
            self.steps_per_episode.append(step)
            
            # Giảm tỷ lệ khám phá một cách thông minh
            self.adaptive_exploration_decay()
            
            # Hiển thị tiến độ
            if verbose and (episode + 1) % 10 == 0:
                success_rate = np.mean(self.episode_success[-100:]) if len(self.episode_success) >= 100 else np.mean(self.episode_success)
                print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {step}, "
                      f"Epsilon: {self.epsilon:.4f}, Success rate: {success_rate:.2%}")
        
        # Trả về kết quả huấn luyện
        return {
            "rewards": self.episode_rewards,
            "steps": self.steps_per_episode,
            "success_rate": self.episode_success
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