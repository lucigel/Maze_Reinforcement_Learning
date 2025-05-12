# config.py

"""
Cấu hình và các hằng số cho dự án học tăng cường trên mê cung.
"""

# Cấu hình chung
PROJECT_NAME = "Maze Reinforcement Learning"
VERSION = "1.0.0"

# Thư mục
MODEL_DIR = "models"
Q_LEARNING_MODEL_DIR = f"{MODEL_DIR}/q_learning"
SARSA_MODEL_DIR = f"{MODEL_DIR}/sarsa"
MAZE_DIR = "mazes"

# Cấu hình mê cung
MAZE_SIZES = {
    "small": (10, 10),
    "medium": (15, 15),
    "large": (20, 20),
    "xlarge": (30, 30)
}

# Hằng số mê cung
PATH = 0
WALL = 1
START = 2
GOAL = 3

# Hướng di chuyển
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
ACTION_VECTORS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # (dx, dy) for [UP, RIGHT, DOWN, LEFT]

# Cấu hình môi trường
MOVE_REWARD = -1.0       # Phần thưởng cho mỗi bước đi
WALL_PENALTY = -5.0      # Phạt khi đi vào tường
GOAL_REWARD = 100.0      # Phần thưởng khi đến đích
TIME_PENALTY = -0.1      # Phạt theo thời gian
MAX_STEPS = 1000         # Số bước tối đa trong một episode

# Cấu hình huấn luyện
TRAINING_EPISODES = 1000  # Số episode huấn luyện
EVAL_INTERVAL = 100       # Đánh giá sau bao nhiêu episode
RENDER_INTERVAL = 200     # Hiển thị môi trường sau bao nhiêu episode

# Tham số Q-Learning
LEARNING_RATE = 0.1        # Tốc độ học (alpha)
DISCOUNT_FACTOR = 0.95     # Hệ số giảm (gamma)
EXPLORATION_RATE = 1.0     # Tỷ lệ khám phá ban đầu (epsilon)
EXPLORATION_DECAY = 0.995  # Tốc độ giảm tỷ lệ khám phá
MIN_EXPLORATION = 0.01     # Giá trị nhỏ nhất của tỷ lệ khám phá

# Cấu hình hiển thị
CONSOLE_OUTPUT = True     # Hiển thị trên console
MATPLOTLIB_OUTPUT = True  # Hiển thị bằng matplotlib
FIGSIZE = (10, 10)        # Kích thước figure cho matplotlib

# Cài đặt đánh giá
EVAL_EPISODES = 100       # Số episode để đánh giá
RANDOM_SEED = 42          # Hạt giống ngẫu nhiên cho tính tái tạo

# Các cấu hình cho các thuật toán sinh mê cung
MAZE_GENERATORS = {
    "dfs": "DFSMazeGenerator",
    "prim": "PrimMazeGenerator",
    "wilson": "WilsonMazeGenerator"
}

# Các thuật toán học tăng cường
RL_AGENTS = {
    "q_learning": "QLearningAgent",
    "sarsa": "SARSAAgent"
}

# Định dạng tên file để lưu mô hình
def get_model_filename(agent_type, maze_size, episodes):
    """
    Tạo tên file để lưu mô hình.
    
    Args:
        agent_type (str): Loại agent (q_learning, sarsa)
        maze_size (tuple): Kích thước mê cung (height, width)
        episodes (int): Số episode đã huấn luyện
        
    Returns:
        str: Tên file
    """
    height, width = maze_size
    return f"{agent_type}_maze_{height}x{width}_ep{episodes}.pkl"