# config.py
PROJECT_NAME = "Maze Reinforcement Learning"
VERSION = "1.0.0"

MAZE_SIZES = {
    "small": (11, 11),  # Mê cung 11x11
    "medium": (15, 15), # Mê cung 15x15
    "large": (21, 21),  # Mê cung 21x21
    "xlarge": (31, 31)  # Mê cung 31x31
}

# Tham số học tăng cường
LEARNING_RATE = 0.1         # Tốc độ học cao hơn (Alpha)
DISCOUNT_FACTOR = 0.99      # Hệ số giảm cao hơn (Gamma)
EXPLORATION_RATE = 1.0      # Tỷ lệ khám phá ban đầu (Epsilon)
EXPLORATION_DECAY = 0.995   # Giảm chậm hơn
MIN_EXPLORATION = 0.01      # Mức khám phá tối thiểu cao hơn

# Tham số môi trường
MOVE_REWARD = -0.05          # Giảm phạt cho mỗi bước di chuyển
WALL_PENALTY = -2.0         # Giảm phạt khi đi vào tường
GOAL_REWARD = 100.0         # Phần thưởng khi đến đích
TIME_PENALTY = -0.001        # Giảm phạt theo thời gian
MAX_STEPS = 2000            # Tăng số bước tối đa

# Tham số huấn luyện
TRAINING_EPISODES = 2000    # Tăng số episode
EVAL_INTERVAL = 100         # Đánh giá sau mỗi 100 episode

# Đường dẫn mô hình
Q_LEARNING_MODEL_DIR = "./models/q_learning"
SARSA_MODEL_DIR = "./models/sarsa"
DQN_MODEL_DIR = "./models/dqn"

# Hạt giống ngẫu nhiên
RANDOM_SEED = 42

# Tham số mới cho Experience Replay
REPLAY_BUFFER_SIZE = 1000  # Kích thước buffer
REPLAY_BATCH_SIZE = 64 
REPLAY_START_SIZE = 1000    # Kích thước batch
ADAPTIVE_ALPHA = True       # Sử dụng tốc độ học thích ứng
PRIORITIZED_REPLAY = True   # Sử dụng prioritized experience replay     
# Thêm các hằng số cấu hình cho DQN
DQN_HIDDEN_SIZE = 128
DQN_BATCH_SIZE = 64
DQN_BUFFER_SIZE = 100000
DQN_TARGET_UPDATE_FREQ = 1000
DQN_LEARNING_RATE = 0.001  # Mặc định cho DQN

# Tham số cho Double Q-Learning
USE_DOUBLE_Q = True
TARGET_UPDATE_STEPS = 500



def get_model_filename(agent_type, maze_size, episodes):
    height, width = maze_size
    return f"{agent_type}_{height}x{width}_{episodes}ep.pkl"