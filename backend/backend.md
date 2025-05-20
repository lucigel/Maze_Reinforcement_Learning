maze-solver-project/
├── backend/                           # Backend - Python FastAPI
│   ├── maze_generators/               # Thuật toán sinh mê cung
│   │   ├── __init__.py
│   │   ├── base_generator.py          # Lớp cơ sở cho sinh mê cung
│   │   ├── dfs_generator.py           # Thuật toán DFS
│   │   ├── prim_generator.py          # Thuật toán Prim
│   │   ├── wilson_generator.py        # Thuật toán Wilson
│   │ 
│   │
│   ├── rl_agents/                     # Thuật toán học tăng cường
│   │   ├── __init__.py
│   │   ├── base_agent.py              # Lớp cơ sở cho các agent
│   │   ├── q_learning.py              # Thuật toán Q-Learning
|   |   |__ dqn_agent.py               # thuật toán DQN
│   │   └── sarsa.py                   # Thuật toán SARSA
│   │
│   ├── environment/                   # Môi trường mê cung
│   │   ├── __init__.py
│   │   └── maze_env.py                # Định nghĩa môi trường mê cung
│   │
│   ├── utils/                         # Các tiện ích
│   │   ├── __init__.py
│   │   ├── config.py                  # Cấu hình và hằng số
│   │   ├── data_handler.py            # Xử lý dữ liệu (lưu/tải Q-table)
│   │   └── visualization.py           # Hiển thị mê cung và kết quả
│   │
│   ├── models/                        # Lưu mô hình đã huấn luyện
│   │   ├── q_learning/
│   │   │   ├── maze_10x10.pkl         # Q-table cho mê cung 10x10
│   │   │   └── maze_20x20.pkl         # Q-table cho mê cung 20x20
│   │   └── sarsa/
│   │       ├── maze_10x10.pkl
│   │       └── maze_20x20.pkl
│   │
│   ├── api.py                         # API endpoints chính
│   ├── training.py                    # Script huấn luyện mô hình
│   └── evaluation.py                  # Script đánh giá mô hình