# /base_generator.py

from abc import ABC, abstractmethod
import numpy as np 
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt 


class BaseMazeGenerator(ABC):
    """
    Lớp cơ sở trừu tượng cho các thuật toán sinh mê cung.
    
    Các thuật toán sinh mê cung như DFS, Prim, Wilson sẽ kế thừa từ lớp này
    và cài đặt phương thức generate() theo cách riêng.
    """
    PATH = 0
    WALL = 1
    START = 2 
    GOAL = 3
    
    def __init__(self, height: int = 10, width: int = 10, 
                 start_pos: Optional[Tuple[int, int]] = None,
                 goal_pos: Optional[Tuple[int, int]] = None, 
                 seed: Optional[int] = None):
        """
        Khởi tạo bộ sinh mê cung.
        
        Args:
            height (int): Chiều cao của mê cung (số hàng)
            width (int): Chiều rộng của mê cung (số cột)
            start_pos (Tuple[int, int], optional): Vị trí điểm bắt đầu, mặc định là (0, 0)
            goal_pos (Tuple[int, int], optional): Vị trí điểm đích, mặc định là (height-1, width-1)
            seed (int, optional): Hạt giống cho bộ sinh số ngẫu nhiên
        
        """
        self.height = height 
        self.width = width 
        
        self.start_pos = start_pos if start_pos is not None else (0, 0)
        
        self.goal_pos = goal_pos if goal_pos is not None else (height - 1, width - 1)
        
        self.rng = np.random.RandomState(seed)
        
        self.maze = None
        
    
    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Phương thức trừu tượng để sinh ra mê cung.
        Các lớp con phải cài đặt phương thức này.
        
        Returns:
            np.ndarray: Ma trận 2D biểu diễn mê cung đã tạo
        """
        pass 
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Lấy danh sách các ô kề với ô (row, col).
        
        Args:
            row (int): Chỉ số hàng
            col (int): Chỉ số cột
            
        Returns:
            List[Tuple[int, int]]: Danh sách các ô kề hợp lệ
        """
        neighbors = []
        
        # Các hướng: trên, dưới, trái, phải
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc 
            
            if 0 <= new_row < self.height and 0 <= new_col < self.width:
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def is_valid_maze(self) -> bool: 
        
        """
        Kiểm tra mê cung đã tạo có hợp lệ không (có đường đi từ start đến goal).
        
        Returns:
            bool: True nếu mê cung hợp lệ, False nếu không
        
        """
        
        if self.maze is None: 
            return False 
        
        visited = np.zeros((self.height, self.width), dtype=bool)
        
        queue = [self.start_pos]
        
        visited[self.start_pos] = True

        while queue: 
            row, col = queue.pop(0)
            
            if (row, col) == self.goal_pos: 
                return True
            
            for n_row, n_col in self.get_neighbors(row, col): 
                if not visited[n_row, n_col] and self.maze[n_row, n_col] != self.WALL:
                    visited[n_row, n_col] = True
                    queue.append((n_row, n_col))
                    
        return False
    
    def save_maze(self, file_path: str) -> None:
        """
        Lưu mê cung vào file.
        
        Args:
            file_path (str): Đường dẫn đến file
        """
        
        if self.maze is not None:
            np.save(file_path, self.maze)
            
    
    def load_maze(self, file_path: str) -> np.ndarray:
        """
        Tải mê cung từ file.
        
        Args:
            file_path (str): Đường dẫn đến file
            
        Returns:
            np.ndarray: Ma trận mê cung đã tải
        """
        self.maze = np.load(file_path)
        return self.maze
    
    def visualize(self) -> None: 
        """
        Hiển thị mê cung.
        """
        if self.maze is None:
            print("Mê cung chưa được tạo.")
            return
        
        maze_vis = self.maze.copy()
        
        colors = {
            self.PATH: 'white', 
            self.WALL: 'black', 
            self.START: 'green', 
            self.GOAL: 'red'
        }
        
        cmap = plt.cm.colors.ListedColormap([colors[self.PATH], 
                                             colors[self.WALL], 
                                             colors[self.START], 
                                             colors[self.GOAL]])
        
        plt.figure(figsize=(10, 10))
        plt.imshow(maze_vis, cmap=cmap)
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.title('Mê cung')
        plt.xticks(range(self.width))
        plt.yticks(range(self.height))
        plt.show()
    
        