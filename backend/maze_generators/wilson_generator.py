# wilson_generator.py
import numpy as np 
from typing import Tuple, List, Set, Dict
from .base_generator import BaseMazeGenerator


class WilsonMazeGenerator(BaseMazeGenerator): 
    """
    Sinh mê cung bằng thuật toán Wilson.
    
    Thuật toán Wilson sử dụng quá trình "loop-erased random walks" để tạo ra mê cung:
    1. Bắt đầu với một lưới đầy tường (cells toàn số 1)
    2. Chọn một ô ngẫu nhiên và đánh dấu là đã thăm
    3. Lặp lại cho đến khi tất cả các ô đều được thăm:
       a. Chọn một ô ngẫu nhiên chưa thăm làm ô bắt đầu
       b. Thực hiện đi ngẫu nhiên từ ô bắt đầu, xóa các vòng lặp cho đến khi đi đến một ô đã thăm
       c. Thêm đường đi vừa tạo (không có vòng lặp) vào mê cung và đánh dấu các ô trên đường đi là đã thăm
    """
    
    def __init__(self, height: int = 10, width: int = 10,
                 start_pos: Tuple[int, int] = None, goal_pos: Tuple[int, int] = None, 
                 seed: int = None):
        """
        Khởi tạo bộ sinh mê cung Wilson.
        
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
        Sinh mê cung bằng thuật toán Wilson.
        
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
                
        # Các hướng di chuyển có thể: lên, xuống, trái, phải
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        
        # Chọn một ô bắt đầu ngẫu nhiên và đánh dấu là đã thăm
        first_cell = cells[self.rng.randint(0, len(cells))]
        self.maze[first_cell] = self.PATH
        
        # Tập hợp các ô đã thăm
        visited = {first_cell}
        
        # Lặp cho đến khi tất cả các ô đều được thăm
        while len(visited) < len(cells):
            # Chọn một ô chưa thăm làm điểm bắt đầu đi
            unvisited = [cell for cell in cells if cell not in visited]
            current = unvisited[self.rng.randint(0, len(unvisited))]
            
            # Thực hiện đi ngẫu nhiên
            path = [current]
            path_dict = {}  # Dict lưu hướng đi từ mỗi ô
            
            while current not in visited:
                # Tìm các ô kề có thể đi được
                neighbors = []
                for dr, dc in directions:
                    new_row, new_col = current[0] + dr, current[1] + dc
                    if 0 <= new_row < self.height and 0 <= new_col < self.width:
                        # Chỉ xét các ô có tọa độ lẻ (các ô có thể đi được)
                        if new_row % 2 == 1 and new_col % 2 == 1:
                            neighbors.append((new_row, new_col))
                
                if not neighbors:
                    break
                    
                # Chọn ngẫu nhiên một hướng đi
                next_cell = neighbors[self.rng.randint(0, len(neighbors))]
                
                # Lưu hướng đi từ ô hiện tại
                path_dict[current] = next_cell
                
                # Cập nhật ô hiện tại
                current = next_cell
                
                # Nếu đã đi qua ô này trước đó trong đường đi hiện tại, xóa vòng lặp
                if current in path:
                    idx = path.index(current)
                    path = path[:idx + 1]
                else:
                    path.append(current)
            
            # Nếu đã đi đến một ô đã thăm, thêm đường đi vào mê cung
            if current in visited:
                # Bắt đầu từ ô đầu tiên của đường đi và đi theo path_dict
                current = path[0]
                while current not in visited:
                    # Đánh dấu ô hiện tại là đã thăm
                    self.maze[current] = self.PATH
                    visited.add(current)
                    
                    # Lấy ô tiếp theo trong đường đi
                    next_cell = path_dict[current]
                    
                    # Phá tường giữa ô hiện tại và ô tiếp theo
                    wall_row = (current[0] + next_cell[0]) // 2
                    wall_col = (current[1] + next_cell[1]) // 2
                    self.maze[wall_row, wall_col] = self.PATH
                    
                    # Cập nhật ô hiện tại
                    current = next_cell
        
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