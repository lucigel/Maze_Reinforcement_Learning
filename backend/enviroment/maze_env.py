# maze_env.py
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from maze_generators.base_generator import BaseMazeGenerator


class MazeEnvironment:
    """
    Môi trường mê cung cho bài toán học tăng cường.
    
    Môi trường này tuân theo giao diện tương tự như OpenAI Gym, với các phương thức
    reset(), step() và render() để agent có thể tương tác.
    """
    PATH = BaseMazeGenerator.PATH
    WALL = BaseMazeGenerator.WALL
    START = BaseMazeGenerator.START
    GOAL = BaseMazeGenerator.GOAL
    
    # Các hành động: 0: lên, 1: xuống, 2: trái, 3: phải,
    ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
    
    def __init__(self, maze: Optional[np.ndarray] = None, 
                 maze_generator: Optional[BaseMazeGenerator] = None,
                 start_pos: Optional[Tuple[int, int]] = None,
                 goal_pos: Optional[Tuple[int, int]] = None,
                 max_steps: int = 1000,
                 move_reward: float = -1.0,
                 wall_penalty: float = -5.0,
                 goal_reward: float = 100.0,
                 time_penalty: float = -0.1):
        """
        Khởi tạo môi trường mê cung.
        
        Args:
            maze (np.ndarray, optional): Ma trận mê cung đã tạo sẵn
            maze_generator (BaseMazeGenerator, optional): Bộ sinh mê cung
            start_pos (Tuple[int, int], optional): Vị trí bắt đầu, mặc định từ maze
            goal_pos (Tuple[int, int], optional): Vị trí đích, mặc định từ maze
            max_steps (int): Số bước tối đa trong một episode
            move_reward (float): Phần thưởng cho mỗi bước di chuyển
            wall_penalty (float): Phạt khi đi vào tường
            goal_reward (float): Phần thưởng khi đến đích
            time_penalty (float): Phạt theo thời gian (tăng theo số bước)
        """
        self.maze_generator = maze_generator
        self.max_steps = max_steps
        self.move_reward = move_reward
        self.wall_penalty = wall_penalty
        self.goal_reward = goal_reward
        self.time_penalty = time_penalty
        
        # Nếu mê cung được cung cấp trực tiếp
        if maze is not None:
            self.maze = maze
            self.height, self.width = maze.shape
            
            # Tìm điểm bắt đầu và kết thúc từ mê cung nếu không được chỉ định
            if start_pos is None:
                start_positions = np.where(self.maze == self.START)
                if len(start_positions[0]) > 0:
                    self.start_pos = (start_positions[0][0], start_positions[1][0])
                else:
                    self.start_pos = (0, 0)  # Mặc định
            else:
                self.start_pos = start_pos
            
            if goal_pos is None:
                goal_positions = np.where(self.maze == self.GOAL)
                if len(goal_positions[0]) > 0:
                    self.goal_pos = (goal_positions[0][0], goal_positions[1][0])
                else:
                    self.goal_pos = (self.height - 1, self.width - 1)  # Mặc định
            else:
                self.goal_pos = goal_pos
        
        # Nếu được cung cấp bộ sinh mê cung
        elif maze_generator is not None:
            self.maze = maze_generator.generate()
            self.height, self.width = self.maze.shape
            self.start_pos = maze_generator.start_pos
            self.goal_pos = maze_generator.goal_pos
        
        else:
            raise ValueError("Phải cung cấp ma trận mê cung hoặc bộ sinh mê cung")
        
        # Biến trạng thái
        self.current_pos = self.start_pos
        self.steps_count = 0
        self.done = False
        
        # Kiểm tra tính hợp lệ của mê cung
        if not self._is_valid_maze():
            print("Cảnh báo: Không tìm thấy đường đi từ điểm bắt đầu đến đích.")
    
    def reset(self) -> Tuple[int, int]:
        """
        Khởi tạo lại môi trường, trả về trạng thái ban đầu.
        
        Returns:
            Tuple[int, int]: Trạng thái ban đầu (vị trí bắt đầu)
        """
        self.current_pos = self.start_pos
        self.steps_count = 0
        self.done = False
        return self.current_pos
    

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        self.steps_count += 1
        
        # Lưu vị trí trước khi di chuyển để tính phần thưởng dựa trên tiến độ
        prev_pos = self.current_pos
        prev_dist_to_goal = self._manhattan_distance(prev_pos, self.goal_pos)
        
        dx, dy = self.ACTIONS[action]
        new_row = self.current_pos[0] + dx
        new_col = self.current_pos[1] + dy
        new_pos = (new_row, new_col)
        
        # Kiểm tra nếu vị trí mới hợp lệ
        if not self._is_valid_pos(new_row, new_col):
            reward = self.wall_penalty
            info = {"status": "hit_wall", "action": self.ACTION_NAMES[action]}
        else:
            self.current_pos = new_pos
            
            if self.current_pos == self.goal_pos:
                reward = self.goal_reward
                self.done = True
                info = {"status": "reached_goal", "action": self.ACTION_NAMES[action]}
            else:
                # Phần thưởng cơ bản cho mỗi bước di chuyển
                reward = self.move_reward
                
                # Phần thưởng dựa trên tiến độ hướng tới mục tiêu
                current_dist_to_goal = self._manhattan_distance(self.current_pos, self.goal_pos)
                progress_reward = prev_dist_to_goal - current_dist_to_goal
                
                # Thưởng cho tiến gần hơn đến mục tiêu, phạt cho xa hơn
                reward += progress_reward * 0.5
                
                # Phần thưởng nhỏ cho việc khám phá các ô mới
                if hasattr(self, 'visited_cells') and self.current_pos not in self.visited_cells:
                    reward += 0.2
                
                if hasattr(self, 'visited_cells'):
                    self.visited_cells.add(self.current_pos)
                else:
                    self.visited_cells = {self.current_pos}
                    
                info = {"status": "moving", "action": self.ACTION_NAMES[action]}
        
        if self.steps_count >= self.max_steps and not self.done:
            self.done = True
            info["status"] = "max_steps_reached"
        
        return self.current_pos, reward, self.done, info

    def _manhattan_distance(self, pos1, pos2):
        """Tính khoảng cách Manhattan giữa hai vị trí"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def _is_valid_pos(self, row: int, col: int) -> bool:
        """
        Kiểm tra nếu vị trí (row, col) là hợp lệ (không phải tường và nằm trong mê cung).
        
        Args:
            row (int): Chỉ số hàng
            col (int): Chỉ số cột
            
        Returns:
            bool: True nếu vị trí hợp lệ, False nếu không
        """
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return self.maze[row, col] != self.WALL
    
    def _is_valid_maze(self) -> bool:
        """
        Kiểm tra mê cung có hợp lệ không (có đường đi từ điểm bắt đầu đến đích).
        
        Returns:
            bool: True nếu mê cung hợp lệ, False nếu không
        """
        visited = np.zeros((self.height, self.width), dtype=bool)
        queue = [self.start_pos]
        visited[self.start_pos] = True
        
        while queue:
            row, col = queue.pop(0)
            
            if (row, col) == self.goal_pos:
                return True
            
            for dx, dy in self.ACTIONS:
                new_row, new_col = row + dx, col + dy
                
                if 0 <= new_row < self.height and 0 <= new_col < self.width and \
                   not visited[new_row, new_col] and self.maze[new_row, new_col] != self.WALL:
                    visited[new_row, new_col] = True
                    queue.append((new_row, new_col))
        
        return False
    
    def get_state_size(self) -> Tuple[int, int]:
        """
        Lấy kích thước không gian trạng thái.
        
        Returns:
            Tuple[int, int]: (height, width) của mê cung
        """
        return (self.height, self.width)
    
    def get_action_size(self) -> int:
        """
        Lấy kích thước không gian hành động.
        
        Returns:
            int: Số lượng hành động có thể thực hiện
        """
        return len(self.ACTIONS)
    
    def get_possible_actions(self, state: Tuple[int, int]) -> List[int]:
        """
        Lấy danh sách các hành động hợp lệ từ trạng thái cụ thể.
        
        Args:
            state (Tuple[int, int]): Trạng thái hiện tại (row, col)
            
        Returns:
            List[int]: Danh sách các hành động hợp lệ
        """
        row, col = state
        valid_actions = []
        
        for action, (dx, dy) in enumerate(self.ACTIONS):
            new_row, new_col = row + dx, col + dy
            
            if self._is_valid_pos(new_row, new_col):
                valid_actions.append(action)
        
        return valid_actions
    
    def render(self, mode: str = 'console') -> None:
        """
        Hiển thị trạng thái hiện tại của môi trường.
        
        Args:
            mode (str): Chế độ hiển thị ('console' hoặc 'matplotlib')
        """
        if mode == 'console':
            # Tạo bản sao ma trận mê cung để hiển thị
            render_maze = np.copy(self.maze)
            
            # Đánh dấu vị trí hiện tại (nếu không phải điểm bắt đầu hoặc đích)
            if self.current_pos != self.start_pos and self.current_pos != self.goal_pos:
                render_maze[self.current_pos] = 4  # 4 là mã cho vị trí hiện tại
            
            # In ra console
            for row in range(self.height):
                for col in range(self.width):
                    cell = render_maze[row, col]
                    if cell == self.WALL:
                        print("█", end="")
                    elif cell == self.START:
                        print("S", end="")
                    elif cell == self.GOAL:
                        print("G", end="")
                    elif (row, col) == self.current_pos:
                        print("A", end="")
                    else:
                        print(" ", end="")
                print()
            print(f"Bước: {self.steps_count}, Vị trí: {self.current_pos}")
            
        elif mode == 'matplotlib':
            import matplotlib.pyplot as plt
            
            # Tạo bản sao ma trận mê cung để hiển thị
            render_maze = np.copy(self.maze)
            
            # Đánh dấu vị trí hiện tại (nếu không phải điểm bắt đầu hoặc đích)
            if self.current_pos != self.start_pos and self.current_pos != self.goal_pos:
                render_maze[self.current_pos] = 4  # 4 là mã cho vị trí hiện tại
            
            colors = {
                self.PATH: 'white',
                self.WALL: 'black',
                self.START: 'green',
                self.GOAL: 'red',
                4: 'blue'  # Vị trí hiện tại
            }
            
            cmap = plt.cm.colors.ListedColormap([colors[self.PATH], 
                                                colors[self.WALL], 
                                                colors[self.START], 
                                                colors[self.GOAL],
                                                colors[4]])
            
            plt.figure(figsize=(10, 10))
            plt.imshow(render_maze, cmap=cmap)
            plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
            plt.title(f'Mê cung - Bước: {self.steps_count}')
            plt.xticks(range(self.width))
            plt.yticks(range(self.height))
            plt.show()
        else:
            raise ValueError(f"Chế độ hiển thị không hợp lệ: {mode}")
    
    def get_shortest_path(self) -> List[Tuple[int, int]]:
        """
        Tìm đường đi ngắn nhất từ điểm bắt đầu đến đích sử dụng BFS.
        
        Returns:
            List[Tuple[int, int]]: Danh sách các vị trí trên đường đi ngắn nhất
        """
        if not self._is_valid_maze():
            return []
        
        # Sử dụng BFS để tìm đường đi ngắn nhất
        queue = [(self.start_pos, [self.start_pos])]
        visited = {self.start_pos}
        
        while queue:
            (row, col), path = queue.pop(0)
            
            if (row, col) == self.goal_pos:
                return path
            
            for dx, dy in self.ACTIONS:
                new_row, new_col = row + dx, col + dy
                new_pos = (new_row, new_col)
                
                if self._is_valid_pos(new_row, new_col) and new_pos not in visited:
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))
        
        return []
    
    def get_optimal_policy(self) -> np.ndarray:
        """
        Tính toán chính sách tối ưu sử dụng quy hoạch động (value iteration).
        
        Returns:
            np.ndarray: Ma trận các hành động tối ưu cho mỗi vị trí
        """
        # Khởi tạo ma trận giá trị V và ma trận chính sách
        V = np.zeros((self.height, self.width))
        policy = np.zeros((self.height, self.width), dtype=int)
        
        # Các tham số
        gamma = 0.99  # Hệ số giảm
        theta = 1e-6  # Ngưỡng hội tụ
        max_iterations = 1000
        
        # Value Iteration
        for _ in range(max_iterations):
            delta = 0
            
            for row in range(self.height):
                for col in range(self.width):
                    if (row, col) == self.goal_pos:
                        V[row, col] = self.goal_reward
                        continue
                    
                    if self.maze[row, col] == self.WALL:
                        continue
                    
                    v = V[row, col]
                    values = []
                    
                    for action in range(len(self.ACTIONS)):
                        dx, dy = self.ACTIONS[action]
                        new_row, new_col = row + dx, col + dy
                        
                        if self._is_valid_pos(new_row, new_col):
                            # Nếu là đích
                            if (new_row, new_col) == self.goal_pos:
                                values.append(self.move_reward + gamma * self.goal_reward)
                            else:
                                values.append(self.move_reward + gamma * V[new_row, new_col])
                        else:
                            values.append(self.wall_penalty + gamma * V[row, col])
                    
                    # Chọn hành động tốt nhất
                    if values:
                        best_value = max(values)
                        best_action = values.index(best_value)
                        
                        V[row, col] = best_value
                        policy[row, col] = best_action
                        
                        delta = max(delta, abs(v - best_value))
            
            if delta < theta:
                break
        
        return policy