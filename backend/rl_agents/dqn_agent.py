# dqn_agent.py (Enhanced Version)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from typing import Tuple, List, Dict, Any, Optional
from rl_agents.base_agent import BaseAgent  
from collections import deque, namedtuple


Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNetwork(nn.Module):
    """
    Mạng neural cải tiến dùng cho Deep Q-Network.
    
    Sử dụng mạng neural với nhiều lớp hơn và layer normalization để cải thiện
    tốc độ học và ổn định hội tụ.
    """
    
    def __init__(self, state_dim: int, action_size: int, hidden_size: int = 128):
        """
        Khởi tạo mạng DQN cải tiến.
        
        Args:
            state_dim (int): Số chiều của vector trạng thái đầu vào
            action_size (int): Số lượng hành động có thể thực hiện
            hidden_size (int): Kích thước của các lớp ẩn
        """
        super(DQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # Layer normalization cho ổn định
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        # Khởi tạo trọng số theo cách giúp đẩy nhanh quá trình hội tụ
        self._init_weights()
    
    def _init_weights(self):
        """Khởi tạo trọng số để cải thiện tốc độ học"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming Initialization phù hợp với ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # Khởi tạo bias nhỏ dương
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x):
        """
        Chuyển tiếp qua mạng neural.
        
        Args:
            x (torch.Tensor): Tensor đầu vào biểu diễn trạng thái
            
        Returns:
            torch.Tensor: Giá trị Q cho mỗi hành động
        """
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay cho DQN.
    Ưu tiên các trải nghiệm có lỗi TD cao để tăng tốc quá trình học.
    """
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Khởi tạo Prioritized Replay Buffer.
        
        Args:
            capacity (int): Kích thước tối đa của buffer
            alpha (float): Hệ số quyết định mức độ ưu tiên dựa trên TD error
            beta_start (float): Hệ số ban đầu để giảm thiểu bias trong sampling
            beta_frames (int): Số frame để tăng beta từ beta_start đến 1
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1  # Để tính beta
    
    def beta(self):
        """Tính beta dựa trên frame hiện tại"""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def push(self, *args):
        """Thêm trải nghiệm vào buffer với ưu tiên cao nhất"""
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(*args))
        else:
            self.buffer[self.pos] = Experience(*args)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        """Lấy mẫu dựa trên ưu tiên"""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        # Tính xác suất lấy mẫu dựa trên ưu tiên
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # Lấy mẫu theo xác suất
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Tính importance-sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta())
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        
        self.frame += 1
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Cập nhật ưu tiên dựa trên TD errors mới"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent(BaseAgent):
    """
    Triển khai thuật toán Deep Q-Network (DQN) được cải tiến cho bài toán mê cung.
    
    Cải tiến bao gồm:
    - Double DQN: sử dụng 2 mạng để giảm overestimation bias
    - Prioritized Experience Replay: ưu tiên học từ trải nghiệm quan trọng
    - N-step returns: học từ phần thưởng nhiều bước
    - Adaptive exploration: điều chỉnh exploration dựa trên hiệu suất
    """
    
    def __init__(self, state_size: Tuple[int, int], action_size: int = 4, 
                 learning_rate: float = 0.001, discount_factor: float = 0.99, 
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995,
                 min_exploration_rate: float = 0.05, seed: Optional[int] = None,
                 buffer_size: int = 100000, batch_size: int = 64,
                 target_update_freq: int = 500, hidden_size: int = 256,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 double_dqn: bool = True, n_step: int = 3,
                 prioritized_replay: bool = True, alpha: float = 0.6,
                 beta_start: float = 0.4):
        """
        Khởi tạo agent DQN cải tiến.
        
        Args:
            state_size (Tuple[int, int]): Kích thước không gian trạng thái (height, width)
            action_size (int): Số lượng hành động có thể thực hiện
            learning_rate (float): Tốc độ học cho optimizer
            discount_factor (float): Hệ số giảm (gamma)
            exploration_rate (float): Tỷ lệ khám phá ban đầu (epsilon)
            exploration_decay (float): Tốc độ giảm tỷ lệ khám phá
            min_exploration_rate (float): Giá trị nhỏ nhất của tỷ lệ khám phá
            seed (int, optional): Hạt giống cho bộ sinh số ngẫu nhiên
            buffer_size (int): Kích thước buffer cho experience replay
            batch_size (int): Kích thước batch cho việc học
            target_update_freq (int): Tần suất cập nhật mạng target (tính theo steps)
            hidden_size (int): Kích thước của các lớp ẩn trong mạng neural
            device (str): Thiết bị sử dụng cho Pytorch (cuda hoặc cpu)
            double_dqn (bool): Sử dụng Double DQN thay vì DQN thông thường
            n_step (int): Số bước cho n-step returns
            prioritized_replay (bool): Sử dụng Prioritized Experience Replay
            alpha (float): Hệ số ưu tiên trong PER
            beta_start (float): Giá trị beta ban đầu trong PER
        """
        super().__init__(state_size, action_size, learning_rate, discount_factor,
                       exploration_rate, exploration_decay, min_exploration_rate, seed, buffer_size)
        
        # Thiết lập cho PyTorch
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        
        # Kích thước đầu vào state cho mạng neural
        self.state_dim = state_size[0] * state_size[1] + 2  # Thêm 2 chiều cho biểu diễn vị trí đích
        
        # Tham số học
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.n_step = n_step
        self.use_prioritized_replay = prioritized_replay
        
        # Tạo mạng neural
        self.policy_net = DQNetwork(self.state_dim, action_size, hidden_size).to(self.device)
        self.target_net = DQNetwork(self.state_dim, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network chỉ dùng để dự đoán
        
        # Optimizer với learning rate decay
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Replay memory
        if self.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size, alpha, beta_start)
        else:
            self.memory = deque(maxlen=buffer_size)
        
        # N-step returns
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Theo dõi số bước học
        self.steps_done = 0
        self.target_update_freq = target_update_freq
        
        # Thống kê phụ
        self.loss_history = []
        self.episode_success = []
        
        # Lưu thông tin goal position nếu có
        self.goal_pos = None
    
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
        
        # Chiến lược 1: Khai thác (exploitation) với xác suất (1 - epsilon)
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
        
        # Chiến lược 5: Upper Confidence Bound (UCB) khám phá
        if self.rng.random() < 0.4:
            ucb_values = np.zeros(self.action_size)
            total_visits = max(1, np.sum(self.state_action_counts[row, col]))
            
            for a in range(self.action_size):
                # Tránh hành động không hợp lệ
                if a not in valid_actions:
                    ucb_values[a] = float('-inf')
                    continue
                
                # Tính confidence bound
                action_visits = max(1, self.state_action_counts[row, col, a])
                exploration_bonus = self.ucb_c * np.sqrt(np.log(total_visits) / action_visits)
                ucb_values[a] = self.q_table[row, col, a] + exploration_bonus
            
            # Chọn hành động có UCB cao nhất
            return np.argmax(ucb_values)
        
        # Chiến lược 6: Khám phá ngẫu nhiên đơn thuần
        if valid_actions:
            return self.rng.choice(valid_actions)
        else:
            return self.rng.randint(0, self.action_size)
    
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
            if visit_count == 0:  # Trạng thái chưa từng thăm
                novelty_reward = 2.0  # Thưởng lớn cho trạng thái hoàn toàn mới
            else:
                novelty_reward = 1.0 / visit_count
            
            # Phạt nếu quay lại trạng thái gần đây (tránh lặp)
            loop_penalty = 0
            if next_state in self.recent_states[-3:]:  # Kiểm tra 3 trạng thái gần nhất
                loop_penalty = -0.3
            
            # Thêm vào phần thưởng tổng
            total_reward += self.intrinsic_reward_scale * (novelty_reward + loop_penalty)
        
        # Tăng số lần thực hiện hành động này từ trạng thái này
        self.action_counts[state[0], state[1], action] += 1
        self.state_action_counts[state[0], state[1], action] += 1
        
        # Lấy giá trị Q hiện tại cho cặp (state, action)
        current_q = self.q_table[state[0], state[1], action]
        
        # Tính toán giá trị Q mới
        if done:
            # Nếu đã kết thúc episode, không có trạng thái tiếp theo
            target_q = total_reward
        else:
            if self.use_double_q:
                # Double Q-Learning: Chọn action từ Q chính, nhưng lấy giá trị từ Q target
                best_action = np.argmax(self.q_table[next_state[0], next_state[1]])
                max_next_q = self.target_q_table[next_state[0], next_state[1], best_action]
            else:
                # Q-Learning tiêu chuẩn
                max_next_q = np.max(self.q_table[next_state[0], next_state[1]])
            
            # Công thức Q-Learning
            target_q = total_reward + self.gamma * max_next_q
        
        # Tính TD error
        td_error = target_q - current_q
        
        # Tốc độ học thích ứng (giảm khi trạng thái được thăm nhiều lần)
        adaptive_lr = max(0.05, self.lr / (1 + 0.05 * self.state_visits[state]))
        
        # Cập nhật Q-table
        self.q_table[state[0], state[1], action] += adaptive_lr * td_error
        
        # Lưu TD error cho prioritized replay
        self.td_errors.append((state, action, reward, next_state, done, abs(td_error)))
        if len(self.td_errors) > 10000:
            self.td_errors = self.td_errors[-10000:]  # Giữ độ dài buffer ở mức hợp lý
        
        # Cập nhật target network nếu sử dụng Double Q-Learning
        if self.use_double_q:
            self.steps += 1
            if self.steps % self.update_target_steps == 0:
                self.target_q_table = self.q_table.copy()
        
        # Lưu trải nghiệm vào buffer
        self.add_experience((state, action, reward, next_state, done))
    
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
                self.epsilon = max(self.min_epsilon, self.epsilon * 0.8)
            elif recent_success > 0.3:
                # Hiệu suất trung bình, giảm epsilon bình thường
                self.epsilon = max(self.min_epsilon, self.epsilon * 0.9)
            else:
                # Hiệu suất kém, giảm epsilon chậm hơn
                self.epsilon = max(self.min_epsilon, self.epsilon * 0.95)
        else:
            # Chưa đủ dữ liệu, sử dụng decay mặc định
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def train(self, env, num_episodes: int = 1000, max_steps: int = 1000,
              verbose: bool = True, save_path: Optional[str] = None,
              save_interval: int = 100, replay_batch_size: int = 32) -> Dict[str, List]:
        """
        Huấn luyện agent sử dụng thuật toán Q-Learning cải tiến.
        
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
            
            # Đếm số trạng thái đã thăm trong episode này
            visited_states = set()
            stuck_counter = 0  # Đếm số bước bị kẹt
            
            while step < max_steps:
                # Chọn hành động
                action = self.choose_action(state)
                
                # Thực hiện hành động
                next_state, reward, done, info = env.step(action)
                
                # Theo dõi trạng thái đã thăm để phát hiện bị kẹt
                state_key = (next_state[0], next_state[1])
                if state_key in visited_states:
                    stuck_counter += 1
                else:
                    visited_states.add(state_key)
                    stuck_counter = 0
                
                # Nếu agent bị kẹt quá lâu, tăng epsilon tạm thời để thúc đẩy khám phá
                if stuck_counter > 30:  # Ngưỡng thấp hơn để phản ứng nhanh hơn
                    self.epsilon = min(1.0, self.epsilon * 2)
                    # Xóa một phần experience buffer để quên đi các trải nghiệm không tốt
                    if len(self.experience_buffer) > 100:
                        self.experience_buffer = self.experience_buffer[-100:]
                    stuck_counter = 0
                
                # Học từ trải nghiệm
                self.learn(state, action, reward, next_state, done)
                
                # Experience replay mỗi 5 bước
                if step % 5 == 0 and len(self.td_errors) > replay_batch_size:
                    self.prioritized_replay(batch_size=min(replay_batch_size*2, len(self.td_errors)//2))
                
                # Cập nhật trạng thái
                state = next_state
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
            
            # Lưu mô hình định kỳ
            if save_path and (episode + 1) % save_interval == 0:
                self.save_model(f"{save_path}/q_learning_episode_{episode + 1}.pkl")
        
        # Lưu mô hình cuối cùng
        if save_path:
            self.save_model(f"{save_path}/q_learning_final.pkl")
        
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