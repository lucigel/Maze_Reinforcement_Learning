# prim_generator.py
import numpy as np 
from typing import Tuple, List, Set
from .base_generator import BaseMazeGenerator


class PrimMazeGenerator(BaseMazeGenerator): 
    """
    Sinh mê cung bằng thuật toán Prim.
    
    Thuật toán Prim tạo ra mê cung như sau:
    1. Bắt đầu với một lưới đầy tường (cells toàn số 1)
    2. Chọn một ô ngẫu nhiên, đánh dấu là đã thăm và thêm tất cả các ô kề vào danh sách biên (frontier)
    3. Lặp cho đến khi danh sách biên trống:
       a. Chọn ngẫu nhiên một ô từ danh sách biên
       b. Tìm các ô đã thăm kề với ô này
       c. Chọn ngẫu nhiên một ô đã thăm và phá tường giữa hai ô
       d. Đánh dấu ô hiện tại là đã thăm và thêm các ô kề chưa thăm vào danh sách biên
    """
    
    def __init__(self, height: int = 10, width: int = 10,
                 start_pos: Tuple[int, int] = None, goal_pos: Tuple[int, int] = None, 
                 seed: int = None):
        """
        Khởi tạo bộ sinh mê cung Prim.
        
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
        Sinh mê cung bằng thuật toán Prim.
        
        Returns:
            np.ndarray: Ma trận 2D biểu diễn mê cung đã tạo
        """
        
        # Khởi tạo mê cung toàn tường
        self.maze = np.ones((self.height, self.width), dtype=int)
        
        # Tạo danh sách các ô có thể đi qua (ô có tọa độ lẻ)
        cells = []
        for r in range(1, self.height, 2): 
            for c in range(1, self.width, 2): 
                cells.append((r, c))
        
        # Chọn một ô bắt đầu ngẫu nhiên
        start_cell = cells[self.rng.randint(0, len(cells))]
        start_row, start_col = start_cell
        
        # Đánh dấu ô bắt đầu là đường đi
        self.maze[start_row, start_col] = self.PATH
        
        # Danh sách các ô biên (frontier)
        frontier = []
        
        # Thêm các ô kề với ô bắt đầu vào danh sách biên
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]  # up, down, left, right
        
        for dr, dc in directions:
            new_row, new_col = start_row + dr, start_col + dc
            
            if 0 <= new_row < self.height and 0 <= new_col < self.width:
                frontier.append((new_row, new_col, start_row, start_col))
        
        # Lặp đến khi danh sách biên trống
        while frontier:
            # Chọn ngẫu nhiên một ô từ danh sách biên
            idx = self.rng.randint(0, len(frontier))
            cell_row, cell_col, parent_row, parent_col = frontier.pop(idx)
            
            # Nếu ô đã là đường đi, bỏ qua
            if self.maze[cell_row, cell_col] == self.PATH:
                continue
            
            # Đánh dấu ô hiện tại là đường đi
            self.maze[cell_row, cell_col] = self.PATH
            
            # Phá tường giữa ô hiện tại và ô cha
            wall_row = (cell_row + parent_row) // 2
            wall_col = (cell_col + parent_col) // 2
            self.maze[wall_row, wall_col] = self.PATH
            
            # Thêm các ô kề chưa thăm vào danh sách biên
            for dr, dc in directions:
                new_row, new_col = cell_row + dr, cell_col + dc
                
                if 0 <= new_row < self.height and 0 <= new_col < self.width and self.maze[new_row, new_col] == self.WALL:
                    # Chỉ thêm các ô có tọa độ lẻ (các ô có thể đi được)
                    if new_row % 2 == 1 and new_col % 2 == 1:
                        frontier.append((new_row, new_col, cell_row, cell_col))
        
        # Đặt điểm bắt đầu và điểm đích
        start_row, start_col = self.start_pos
        
        if self.maze[start_row, start_col] == self.WALL: 
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