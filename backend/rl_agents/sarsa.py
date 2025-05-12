# sarsa.py

import numpy as np
from typing import Tuple, Optional
from backend.rl_agents.base_agent import BaseAgent

class SARSAAgent(BaseAgent):
    """
    Agent học tăng cường sử dụng thuật toán SARSA (State-Action-Reward-State-Action).
    
    SARSA là thuật toán học tăng cường kiểu on-policy, sử dụng chính sách 
    hiện tại để cập nhật Q-table.
    """
    
    def __init__(self, state_size: Tuple[int, int], action_size: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995,
                 min_exploration_rate: float = 0.01, seed: Optional[int] = None):
        """
        Khởi tạo SARSA agent.
        
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
        
        # Lưu hành động hiện tại để sử dụng trong quá trình học
        self.current_action = None
    
    def choose_action(self, state: Tuple[int, int]) -> int:
        """
        Chọn hành động dựa trên trạng thái hiện tại, sử dụng chính sách epsilon-greedy.
        
        Args:
            state (Tuple[int, int]): Trạng thái hiện tại (row, col)
            
        Returns:
            int: Hành động được chọn
        """
        row, col = state
        
        # Chiến lược Epsilon-greedy
        if self.rng.random() < self.epsilon:
            # Khám phá: chọn ngẫu nhiên một hành động
            action = self.rng.randint(0, self.action_size)
        else:
            # Khai thác: chọn hành động có giá trị Q cao nhất
            action = np.argmax(self.q_table[row, col])
        
        # Lưu trữ hành động hiện tại để sử dụng trong learn()
        self.current_action = action
        
        return action
    
    def learn(self, state: Tuple[int, int], action: int, 
              reward: float, next_state: Tuple[int, int], 
              done: bool, next_action: Optional[int] = None) -> None:
        """
        Cập nhật Q-table dựa trên trải nghiệm hiện tại.
        
        Công thức cập nhật SARSA:
        Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
        
        Args:
            state (Tuple[int, int]): Trạng thái hiện tại (row, col)
            action (int): Hành động được thực hiện
            reward (float): Phần thưởng nhận được
            next_state (Tuple[int, int]): Trạng thái tiếp theo
            done (bool): True nếu episode kết thúc
            next_action (int, optional): Hành động tiếp theo được chọn từ hàm choose_action
        """
        row, col = state
        next_row, next_col = next_state
        
        # Nếu next_action không được cung cấp, sử dụng hành động đã lưu từ lần gọi choose_action gần nhất
        if next_action is None:
            next_action = self.current_action
        
        # Giá trị Q hiện tại
        current_q = self.q_table[row, col, action]
        
        # Giá trị Q tiếp theo (sử dụng hành động tiếp theo thay vì giá trị lớn nhất như trong Q-learning)
        if done:
            # Nếu là trạng thái kết thúc, không có phần thưởng từ trạng thái tiếp theo
            next_q = 0
        else:
            next_q = self.q_table[next_row, next_col, next_action]
        
        # Tính toán target Q-value theo công thức SARSA
        target_q = reward + self.gamma * next_q
        
        # Cập nhật Q-table
        self.q_table[row, col, action] += self.lr * (target_q - current_q)
    
    def train_episode(self, env, max_steps=1000) -> Tuple[float, int]:
        """
        Huấn luyện agent qua một episode.
        
        Args:
            env: Môi trường mê cung
            max_steps (int): Số bước tối đa trong một episode
            
        Returns:
            Tuple[float, int]: (Tổng phần thưởng, Số bước thực hiện)
        """
        # Reset môi trường
        state = env.reset()
        
        # Chọn hành động đầu tiên
        action = self.choose_action(state)
        
        total_reward = 0
        steps = 0
        done = False
        
        # Vòng lặp cho mỗi bước trong episode
        while not done and steps < max_steps:
            # Thực hiện hành động và nhận kết quả
            next_state, reward, done, _ = env.step(action)
            
            # Chọn hành động tiếp theo từ trạng thái mới (khác với Q-learning)
            next_action = self.choose_action(next_state)
            
            # Cập nhật Q-table
            self.learn(state, action, reward, next_state, done, next_action)
            
            # Chuyển sang trạng thái tiếp theo
            state = next_state
            action = next_action
            
            # Cập nhật tổng phần thưởng và số bước
            total_reward += reward
            steps += 1
        
        # Giảm tỷ lệ khám phá
        self.decay_exploration()
        
        # Lưu lại thông tin huấn luyện
        self.episode_rewards.append(total_reward)
        self.steps_per_episode.append(steps)
        
        return total_reward, steps