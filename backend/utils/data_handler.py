# data_handler.py

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union

from backend.utils.config import (
    MODEL_DIR, Q_LEARNING_MODEL_DIR, SARSA_MODEL_DIR,
    get_model_filename
)

class DataHandler:
    """
    Lớp xử lý dữ liệu cho việc lưu và tải các mô hình, kết quả huấn luyện, và trực quan hóa.
    """
    
    @staticmethod
    def ensure_dir_exists(directory: str) -> None:
        """
        Đảm bảo thư mục tồn tại, nếu không thì tạo mới.
        
        Args:
            directory (str): Đường dẫn thư mục
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def save_model(agent, agent_type: str, maze_size: Tuple[int, int], episodes: int) -> str:
        """
        Lưu mô hình vào file.
        
        Args:
            agent: Agent học tăng cường (QLearningAgent hoặc SARSAAgent)
            agent_type (str): Loại agent ('q_learning' hoặc 'sarsa')
            maze_size (Tuple[int, int]): Kích thước mê cung
            episodes (int): Số episode đã huấn luyện
            
        Returns:
            str: Đường dẫn đến file đã lưu
        """
        # Chọn thư mục lưu dựa vào loại agent
        if agent_type == 'q_learning':
            model_dir = Q_LEARNING_MODEL_DIR
        elif agent_type == 'sarsa':
            model_dir = SARSA_MODEL_DIR
        else:
            model_dir = MODEL_DIR
        
        # Đảm bảo thư mục tồn tại
        DataHandler.ensure_dir_exists(model_dir)
        
        # Tạo tên file
        filename = get_model_filename(agent_type, maze_size, episodes)
        file_path = os.path.join(model_dir, filename)
        
        # Lưu mô hình
        agent.save_model(file_path)
        
        return file_path
    
    @staticmethod
    def load_model(agent, file_path: str) -> None:
        """
        Tải mô hình từ file.
        
        Args:
            agent: Agent học tăng cường (QLearningAgent hoặc SARSAAgent)
            file_path (str): Đường dẫn đến file
        """
        agent.load_model(file_path)
    
    @staticmethod
    def export_to_json(agent, agent_type: str, maze_size: Tuple[int, int], episodes: int) -> str:
        """
        Xuất mô hình sang định dạng JSON để sử dụng trên web.
        
        Args:
            agent: Agent học tăng cường
            agent_type (str): Loại agent ('q_learning' hoặc 'sarsa')
            maze_size (Tuple[int, int]): Kích thước mê cung
            episodes (int): Số episode đã huấn luyện
            
        Returns:
            str: Đường dẫn đến file JSON
        """
        # Chọn thư mục lưu dựa vào loại agent
        if agent_type == 'q_learning':
            model_dir = os.path.join(Q_LEARNING_MODEL_DIR, 'json')
        elif agent_type == 'sarsa':
            model_dir = os.path.join(SARSA_MODEL_DIR, 'json')
        else:
            model_dir = os.path.join(MODEL_DIR, 'json')
        
        # Đảm bảo thư mục tồn tại
        DataHandler.ensure_dir_exists(model_dir)
        
        # Tạo tên file
        height, width = maze_size
        filename = f"{agent_type}_maze_{height}x{width}_ep{episodes}.json"
        file_path = os.path.join(model_dir, filename)
        
        # Xuất mô hình sang JSON
        agent.export_to_json(file_path)
        
        return file_path
    
    @staticmethod
    def save_maze(maze: np.ndarray, filename: str) -> str:
        """
        Lưu mê cung vào file.
        
        Args:
            maze (np.ndarray): Ma trận mê cung
            filename (str): Tên file
            
        Returns:
            str: Đường dẫn đến file đã lưu
        """
        maze_dir = "mazes"
        DataHandler.ensure_dir_exists(maze_dir)
        
        file_path = os.path.join(maze_dir, filename)
        np.save(file_path, maze)
        
        return file_path
    
    @staticmethod
    def load_maze(file_path: str) -> np.ndarray:
        """
        Tải mê cung từ file.
        
        Args:
            file_path (str): Đường dẫn đến file
            
        Returns:
            np.ndarray: Ma trận mê cung
        """
        return np.load(file_path)
    
    @staticmethod
    def save_training_history(history: Dict[str, List], agent_type: str, 
                             maze_size: Tuple[int, int], episodes: int) -> str:
        """
        Lưu lịch sử huấn luyện vào file.
        
        Args:
            history (Dict[str, List]): Lịch sử huấn luyện (rewards, steps, ...)
            agent_type (str): Loại agent ('q_learning' hoặc 'sarsa')
            maze_size (Tuple[int, int]): Kích thước mê cung
            episodes (int): Số episode đã huấn luyện
            
        Returns:
            str: Đường dẫn đến file đã lưu
        """
        history_dir = os.path.join(MODEL_DIR, 'history')
        DataHandler.ensure_dir_exists(history_dir)
        
        height, width = maze_size
        filename = f"{agent_type}_maze_{height}x{width}_ep{episodes}_history.pkl"
        file_path = os.path.join(history_dir, filename)
        
        with open(file_path, 'wb') as f:
            pickle.dump(history, f)
        
        return file_path
    
    @staticmethod
    def load_training_history(file_path: str) -> Dict[str, List]:
        """
        Tải lịch sử huấn luyện từ file.
        
        Args:
            file_path (str): Đường dẫn đến file
            
        Returns:
            Dict[str, List]: Lịch sử huấn luyện
        """
        with open(file_path, 'rb') as f:
            history = pickle.load(f)
        
        return history
    
    @staticmethod
    def plot_training_progress(history: Dict[str, List], title: str = "Training Progress", 
                              save_path: Optional[str] = None) -> None:
        """
        Vẽ biểu đồ tiến trình huấn luyện.
        
        Args:
            history (Dict[str, List]): Lịch sử huấn luyện
            title (str): Tiêu đề biểu đồ
            save_path (str, optional): Đường dẫn để lưu biểu đồ
        """
        plt.figure(figsize=(15, 6))
        
        # Vẽ biểu đồ phần thưởng
        plt.subplot(1, 3, 1)
        rewards = history.get('rewards', [])
        plt.plot(rewards)
        plt.title('Phần thưởng theo episode')
        plt.xlabel('Episode')
        plt.ylabel('Tổng phần thưởng')
        
        # Vẽ biểu đồ số bước
        plt.subplot(1, 3, 2)
        steps = history.get('steps', [])
        plt.plot(steps)
        plt.title('Số bước theo episode')
        plt.xlabel('Episode')
        plt.ylabel('Số bước')
        
        # Vẽ biểu đồ epsilon (tỷ lệ khám phá)
        plt.subplot(1, 3, 3)
        epsilons = history.get('epsilons', [])
        if epsilons:
            plt.plot(epsilons)
            plt.title('Tỷ lệ khám phá (epsilon)')
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    @staticmethod
    def plot_comparison(histories: Dict[str, Dict[str, List]], 
                       metric: str = 'rewards', 
                       title: str = "So sánh các thuật toán",
                       save_path: Optional[str] = None) -> None:
        """
        Vẽ biểu đồ so sánh các thuật toán.
        
        Args:
            histories (Dict[str, Dict[str, List]]): Dictionary với key là tên thuật toán,
                                                   value là lịch sử huấn luyện
            metric (str): Chỉ số so sánh ('rewards', 'steps', 'epsilons')
            title (str): Tiêu đề biểu đồ
            save_path (str, optional): Đường dẫn để lưu biểu đồ
        """
        plt.figure(figsize=(10, 6))
        
        for algorithm, history in histories.items():
            values = history.get(metric, [])
            plt.plot(values, label=algorithm)
        
        metric_names = {
            'rewards': 'Phần thưởng',
            'steps': 'Số bước',
            'epsilons': 'Tỷ lệ khám phá'
        }
        
        plt.title(f"{title} - {metric_names.get(metric, metric)}")
        plt.xlabel('Episode')
        plt.ylabel(metric_names.get(metric, metric))
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()