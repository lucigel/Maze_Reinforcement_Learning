# Maze Solver với Reinforcement Learning

Dự án này triển khai các thuật toán học tăng cường (Reinforcement Learning) để giải quyết bài toán tìm đường trong mê cung. Dự án bao gồm 3 thuật toán chính: Q-Learning, SARSA và Deep Q-Network (DQN).

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Thuật toán](#thuật-toán)
- [Kết quả](#kết-quả)
- [Tác giả](#tác-giả)

## 🎯 Giới thiệu

Dự án này nghiên cứu và so sánh hiệu quả của các thuật toán học tăng cường trong việc giải quyết bài toán tìm đường trong mê cung. Các thuật toán được triển khai bao gồm:

- **Q-Learning**: Thuật toán off-policy cơ bản sử dụng Q-table
- **SARSA**: Thuật toán on-policy sử dụng Q-table
- **DQN**: Deep Q-Network sử dụng mạng neural để xấp xỉ Q-function

## 📁 Cấu trúc dự án

```
maze-solver-project/
├── backend/                           # Backend - Python
│   ├── maze_generators/               # Thuật toán sinh mê cung
│   │   ├── __init__.py
│   │   ├── base_generator.py          # Lớp cơ sở cho sinh mê cung
│   │   ├── dfs_generator.py           # Thuật toán DFS
│   │   ├── prim_generator.py          # Thuật toán Prim
│   │   └── wilson_generator.py        # Thuật toán Wilson
│   │
│   ├── rl_agents/                     # Thuật toán học tăng cường
│   │   ├── __init__.py
│   │   ├── base_agent.py              # Lớp cơ sở cho các agent
│   │   ├── q_learning.py              # Thuật toán Q-Learning
│   │   ├── dqn_agent.py               # Thuật toán DQN
│   │   └── sarsa.py                   # Thuật toán SARSA
│   │
│   ├── enviroment/                    # Môi trường mê cung
│   │   ├── __init__.py
│   │   └── maze_env.py                # Định nghĩa môi trường mê cung
│   │
│   ├── utils/                         # Các tiện ích
│   │   ├── __init__.py
│   │   ├── config.py                  # Cấu hình và hằng số
│   │   ├── data_handler.py            # Xử lý dữ liệu (lưu/tải model)
│   │   └── visualization.py           # Hiển thị mê cung và kết quả
│   │
│   ├── models/                        # Lưu mô hình đã huấn luyện
│   │   ├── q_learning/                # Mô hình Q-Learning
│   │   ├── sarsa/                     # Mô hình SARSA
│   │   └── dqn/                       # Mô hình DQN
│   │
│   ├── static/                        # Frontend - Giao diện web
│   │   └── index.html                 # Giao diện HTML chính
│   │
│   ├── results/                       # Kết quả đánh giá
│   │   ├── q_learning_heatmaps/       # Heatmap Q-Learning
│   │   ├── sarsa_heatmaps/            # Heatmap SARSA
│   │   ├── dqn_heatmaps/              # Heatmap DQN
│   │   └── comparison/                # Kết quả so sánh
│   │
│   ├── app.py                         # Web server Flask/FastAPI
│   ├── training.py                    # Script huấn luyện mô hình
│   ├── evaluation.py                  # Script đánh giá mô hình
│   └── get_heatmaps.py               # Script tạo heatmap
│
├── requirements.txt                   # Danh sách thư viện cần thiết
└── README.md                         # Tài liệu hướng dẫn
```

## 💻 Yêu cầu hệ thống

- Python 3.8 trở lên
- CUDA (tùy chọn, cho DQN với GPU)
- RAM: tối thiểu 4GB
- Dung lượng đĩa: 1GB

## 🚀 Cài đặt

### 1. Clone dự án

```bash
git clone https://github.com/yourusername/maze-solver-project.git
cd maze-solver-project
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Nội dung file `requirements.txt`:
```txt
numpy==1.24.3
matplotlib==3.7.1
torch==2.0.1
tqdm==4.65.0
seaborn==0.12.2
pandas==2.0.3
flask==3.0.0
flask-cors==4.0.0
```

## 📖 Hướng dẫn sử dụng

### 1. Chạy giao diện web

```bash
# Khởi động web server
python backend/app.py

# Mở trình duyệt và truy cập
http://localhost:8000
```

Giao diện web cho phép:
- Chọn thuật toán (Q-Learning, SARSA, DQN)
- Chọn kích thước mê cung
- Theo dõi quá trình huấn luyện real-time
- Xem visualization của kết quả
- So sánh các thuật toán

### 2. Huấn luyện mô hình (Command Line)

#### Huấn luyện Q-Learning

```bash
# Mê cung nhỏ (11x11)
python backend/training.py --agent q_learning --maze dfs --size small --episodes 2000

# Mê cung vừa (15x15)
python backend/training.py --agent q_learning --maze dfs --size medium --episodes 2000

# Mê cung lớn (21x21)
python backend/training.py --agent q_learning --maze dfs --size large --episodes 3000

# Với tham số tùy chỉnh
python backend/training.py --agent q_learning --maze dfs --size medium --episodes 2000 --lr 0.15 --epsilon 0.8 --decay 0.99
```

#### Huấn luyện SARSA

```bash
# Mê cung vừa (15x15)
python backend/training.py --agent sarsa --maze dfs --size medium --episodes 2000

# Với các loại mê cung khác
python backend/training.py --agent sarsa --maze prim --size medium --episodes 2000
python backend/training.py --agent sarsa --maze wilson --size medium --episodes 2000
```

#### Huấn luyện DQN

```bash
# Mê cung vừa (15x15)
python backend/training.py --agent dqn --maze dfs --size medium --episodes 2000 --hidden-size 128 --batch-size 64

# Mê cung lớn với GPU
python backend/training.py --agent dqn --maze dfs --size large --episodes 5000 --hidden-size 256 --batch-size 128
```

#### Tham số huấn luyện

| Tham số | Mô tả | Giá trị mặc định |
|---------|-------|------------------|
| `--agent` | Thuật toán (q_learning, sarsa, dqn) | q_learning |
| `--maze` | Loại mê cung (dfs, prim, wilson) | dfs |
| `--size` | Kích thước (small, medium, large, xlarge) | small |
| `--episodes` | Số episode huấn luyện | 2000 |
| `--lr` | Learning rate | 0.1 |
| `--gamma` | Discount factor | 0.99 |
| `--epsilon` | Exploration rate | 1.0 |
| `--decay` | Epsilon decay | 0.995 |
| `--render` | Hiển thị quá trình train | False |

### 2. Đánh giá mô hình

```bash
# Đánh giá Q-Learning
python backend/evaluation.py \
    --model_path models/q_learning/q_learning_15x15_2000ep.pkl \
    --model_type q_learning \
    --maze_type dfs \
    --maze_size 15 \
    --num_episodes 100 \
    --output_dir results/q_learning_eval

# Đánh giá SARSA
python backend/evaluation.py \
    --model_path models/sarsa/sarsa_15x15_2000ep.pkl \
    --model_type sarsa \
    --maze_type dfs \
    --maze_size 15 \
    --num_episodes 100 \
    --output_dir results/sarsa_eval

# Đánh giá DQN
python backend/evaluation.py \
    --model_path models/dqn/dqn_15x15_2000ep.pth \
    --model_type dqn \
    --maze_type dfs \
    --maze_size 15 \
    --num_episodes 100 \
    --output_dir results/dqn_eval
```

### 3. So sánh các mô hình

```bash
# So sánh Q-Learning vs SARSA
python backend/evaluation.py \
    --model_path models/q_learning/q_learning_15x15_2000ep.pkl \
    --model_type q_learning \
    --maze_type dfs \
    --maze_size 15 \
    --num_episodes 100 \
    --compare \
    --compare_path models/sarsa/sarsa_15x15_2000ep.pkl \
    --compare_type sarsa \
    --output_dir results/comparison_ql_vs_sarsa

# So sánh Q-Learning vs DQN
python backend/evaluation.py \
    --model_path models/q_learning/q_learning_15x15_2000ep.pkl \
    --model_type q_learning \
    --maze_type dfs \
    --maze_size 15 \
    --num_episodes 100 \
    --compare \
    --compare_path models/dqn/dqn_15x15_2000ep.pth \
    --compare_type dqn \
    --output_dir results/comparison_ql_vs_dqn
```

### 4. Tạo heatmap visualization

```bash
# Tạo heatmap cho tất cả các thuật toán
python backend/get_heatmaps.py

# Output sẽ được lưu trong:
# - results/q_learning_heatmaps/
# - results/sarsa_heatmaps/
# - results/dqn_heatmaps/
```

### 6. Chạy toàn bộ pipeline

**Option 1: Sử dụng giao diện web (Khuyến nghị)**
```bash
python backend/app.py
# Sau đó mở http://localhost:8000 trong trình duyệt
```

**Option 2: Sử dụng script command line**

Tạo file `train_all.sh` (Linux/Mac) hoặc `train_all.bat` (Windows):

**Linux/Mac:**
```bash
#!/bin/bash
# train_all.sh

echo "Training Q-Learning..."
python backend/training.py --agent q_learning --maze dfs --size medium --episodes 2000

echo "Training SARSA..."
python backend/training.py --agent sarsa --maze dfs --size medium --episodes 2000

echo "Training DQN..."
python backend/training.py --agent dqn --maze dfs --size medium --episodes 2000

echo "Creating heatmaps..."
python backend/get_heatmaps.py

echo "Training completed!"
```

**Windows:**
```batch
@echo off
REM train_all.bat

echo Training Q-Learning...
python backend/training.py --agent q_learning --maze dfs --size medium --episodes 2000

echo Training SARSA...
python backend/training.py --agent sarsa --maze dfs --size medium --episodes 2000

echo Training DQN...
python backend/training.py --agent dqn --maze dfs --size medium --episodes 2000

echo Creating heatmaps...
python backend/get_heatmaps.py

echo Training completed!
```

## 🧠 Thuật toán

### Q-Learning
- **Loại**: Off-policy
- **Ưu điểm**: Hội tụ nhanh, học từ kinh nghiệm tối ưu
- **Nhược điểm**: Có thể overestimate giá trị Q

### SARSA
- **Loại**: On-policy
- **Ưu điểm**: An toàn hơn, phù hợp với môi trường stochastic
- **Nhược điểm**: Hội tụ chậm hơn Q-Learning

### DQN (Deep Q-Network)
- **Loại**: Off-policy với neural network
- **Ưu điểm**: Xử lý không gian trạng thái lớn, khả năng tổng quát hóa tốt
- **Nhược điểm**: Cần nhiều dữ liệu, tính toán phức tạp

## 📊 Kết quả

Sau khi huấn luyện, các mô hình sẽ được lưu trong thư mục `models/`:
- Q-Learning: `models/q_learning/q_learning_{size}x{size}_{episodes}ep.pkl`
- SARSA: `models/sarsa/sarsa_{size}x{size}_{episodes}ep.pkl`
- DQN: `models/dqn/dqn_{size}x{size}_{episodes}ep.pth`

Kết quả đánh giá sẽ bao gồm:
- Tỷ lệ thành công
- Số bước trung bình
- Phần thưởng trung bình
- Heatmap chính sách và giá trị
- So sánh hiệu suất giữa các thuật toán

## 🔧 Cấu hình

Các tham số cấu hình có thể chỉnh sửa trong `backend/utils/config.py`:

```python
# Kích thước mê cung
MAZE_SIZES = {
    "small": (11, 11),
    "medium": (15, 15),
    "large": (21, 21),
    "xlarge": (31, 31)
}

# Tham số học
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION = 0.01

# Phần thưởng
MOVE_REWARD = -0.05
WALL_PENALTY = -2.0
GOAL_REWARD = 100.0
TIME_PENALTY = -0.001
MAX_STEPS = 2000
```

## 👥 Tác giả

- Tên: NGUYỄN NGỌC DUY
- Email: ngocduy0217@gmail.com

## 📄 License

