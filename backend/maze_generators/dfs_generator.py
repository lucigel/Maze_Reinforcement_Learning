import numpy as np 
from typing import Tuple, List, Set 
from .base_generator import BaseMazeGenerator


class DFSMazeGenerator(BaseMazeGenerator): 
    """
    Sinh mê cung bằng thuật toán Depth-First Search (DFS).
    
    Thuật toán DFS tạo ra mê cung như sau:
    1. Bắt đầu với một lưới đầy tường (cells toàn số 1)
    2. Chọn một ô bắt đầu và đánh dấu là đã thăm
    3. Lặp lại cho đến khi đã thăm hết tất cả các ô:
       a. Nếu ô hiện tại có các ô kề chưa thăm:
          - Chọn ngẫu nhiên một ô kề chưa thăm
          - Phá tường giữa ô hiện tại và ô kề đó
          - Di chuyển đến ô kề và đánh dấu là đã thăm
       b. Nếu không có ô kề chưa thăm, quay lại ô trước đó (backtrack)
    """
    
    def __init__(self, height: int = 10, width: int = 10,
                 start_pos: Tuple[int, int] = None, goal_pos: Tuple[int, int] = None, 
                 seed: int = None):
        """
        Khởi tạo bộ sinh mê cung DFS.
        
        Args:
            height (int): Chiều cao của mê cung (số hàng)
            width (int): Chiều rộng của mê cung (số cột)
            start_pos (Tuple[int, int], optional): Vị trí điểm bắt đầu
            goal_pos (Tuple[int, int], optional): Vị trí điểm đích
            seed (int, optional): Hạt giống cho bộ sinh số ngẫu nhiên
        """
        
        if height % 2 == 0: 
            height += 1 
        if width % 2 == 0: 
            width += 1
            
        super().__init__(height, width, start_pos, goal_pos, seed)
        
    def generate(self) -> np.ndarray:
        """
        Sinh mê cung bằng thuật toán DFS.
        
        Returns:
            np.ndarray: Ma trận 2D biểu diễn mê cung đã tạo
        """
        
        self.maze = np.ones((self.height, self.width), dtype=int)
        
        
        # Tạo danh sách các ô có thể đi qua (ô có tọa độ lẻ)
        # Trong perfect maze, các ô có thể đi qua sẽ có cả tọa độ hàng và cột đều lẻ
        cells = []
        for r in range(1, self.height, 2): 
            for c in range(1, self.width, 2): 
                cells.append((r, c))
                self.maze[r, c] = self.PATH
        
        # Bắt đầu từ một ô ngẫu nhiên
        start_cell = cells[self.rng.randint(0, len(cells))]
        
        visited = {start_cell}
        
        stack = [start_cell]
        
        while stack: 
            current_row, current_col = stack[-1]
            
            neighbors = []
            
            # up, down, left, right
            directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
            
            for dr, dc in directions: 
                new_row, new_col = current_row + dr, current_col + dc
            
                if 0 <= new_row < self.height and 0 <= new_col < self.width and (new_row, new_col) not in visited: 
                
                    if (new_row, new_col) in cells:
                        neighbors.append((new_row, new_col))
                    
            if neighbors: 
                next_cell = neighbors[self.rng.randint(0, len(neighbors))]
                next_row, next_col = next_cell
                
                
                wall_row = current_row + (next_row - current_row) // 2
                wall_col = current_col + (next_col -current_col) // 2
                self.maze[wall_row, wall_col] = self.PATH
                
                stack.append(next_cell)
                visited.add(next_cell)
            else: 
                stack.pop()     
                
        # Đặt điểm bắt đầu và điểm đích
        # Đảm bảo điểm bắt đầu và đích là đường đi, không phải tường       
        start_row, start_col = self.start_pos
        
        if self.maze[start_row, start_col]  == self.WALL: 
            for dr in [-1, 0, 1]: 
                for dc in [-1, 0, 1]: 
                    if 0 <= start_row + dr < self.height and 0 <= start_col + dc < self.width:
                        if self.maze[start_row + dr, start_col + dc] == self.PATH:
                            self.start_pos = (start_row + dr, start_col + dc)
                            break
                        
        
        goal_row, goal_col = self.goal_pos
        if self.maze[goal_row, goal_col] == self.WALL: 
            for dr in [-1, 0, 1]: 
                for dc in [-1, 0, 1]: 
                    if 0 <= goal_row + dr < self.height and 0 <= goal_col + dc < self.width: 
                        if self.maze[goal_row + dr, goal_col + dc] == self.PATH: 
                            self.goal_pos = (goal_row + dr, goal_col + dc)
                            break
                        
        self.maze[self.start_pos] = self.START
        self.maze[self.goal_pos] = self.GOAL
        
        if not self.is_valid_maze(): 
            print("Cảnh báo: Mê cung không có đường đi từ điểm bắt đầu đến điểm đích.")
            
            self._create_path_between(self.start_pos, self.goal_pos)
        
        return self.maze 
    
    def _create_path_between(self, start: Tuple[int, int], end: Tuple[int, int]) -> None: 
        """
        Tạo một đường đi trực tiếp từ điểm bắt đầu đến điểm đích.
        
        Args:
            start (Tuple[int, int]): Điểm bắt đầu
            end (Tuple[int, int]): Điểm đích
        """
        start_row, start_col = start 
        end_row, end_col = end 
        
        row, col = start_row, start_col 
        
        while col != end_col: 
            step = 1 if col < end_col else -1
            col += step 
            if self.maze[row, col] == self.WALL: 
                self.maze[row, col] = self.PATH
        
        while row != end_row: 
            step = 1 if row < end_row else -1 
            row += step 
            if self.maze[row, col] == self.WALL:
                self.maze[row, col] = self.PATH
                
        self.maze[start] = self.START
        self.maze[end] = self.GOAL