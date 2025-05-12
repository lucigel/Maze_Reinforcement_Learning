maze_reinforcement_learning/
│
├── backend/
│   ├── maze_generators/
│   │   ├── __init__.py
│   │   ├── base_generator.py       # Lớp cơ sở cho các bộ sinh mê cung
│   │   ├── dfs_generator.py        # Thuật toán DFS để sinh mê cung
│   │   ├── prim_generator.py       # Thuật toán Prim để sinh mê cung
│   │   └── wilson_generator.py     # Thuật toán Wilson để sinh mê cung
│   │
│   ├── rl_agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Lớp cơ sở cho các agent học tăng cường
│   │   ├── q_learning.py           # Thuật toán Q-Learning
│   │   └── sarsa.py                # Thuật toán SARSA
│   │
│   ├── environment/
│   │   ├── __init__.py
│   │   └── maze_env.py             # Môi trường mê cung
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py               # Cấu hình và hằng số
│   │   ├── data_handler.py         # Xử lý dữ liệu (lưu/tải Q-table)
│   │   └── visualization.py        # Hiển thị mê cung và kết quả (backend)
│   │
│   ├── models/                     # Thư mục lưu các mô hình đã huấn luyện
│   │   ├── q_learning/
│   │   │   ├── maze_10x10.pkl      # Q-table cho mê cung 10x10
│   │   │   └── maze_20x20.pkl      # Q-table cho mê cung 20x20
│   │   └── sarsa/
│   │       ├── maze_10x10.pkl
│   │       └── maze_20x20.pkl
│   │
│   ├── training.py                 # Script để huấn luyện các mô hình
│   ├── evaluation.py               # Script để đánh giá hiệu suất các mô hình
│   └── api.py                      # API để kết nối với frontend (Flask/FastAPI)
│
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   ├── main.css            # Styles chính
│   │   │   └── maze.css            # Styles riêng cho hiển thị mê cung
│   │   │
│   │   ├── js/
│   │   │   ├── main.js             # Logic chính của ứng dụng
│   │   │   ├── mazeRenderer.js     # Hiển thị mê cung trên canvas
│   │   │   ├── agentSimulator.js   # Mô phỏng quá trình giải mê cung của agent
│   │   │   └── api.js              # Xử lý các yêu cầu API đến backend
│   │   │
│   │   └── assets/                 # Hình ảnh và tài nguyên
│   │       ├── agent.png           # Hình ảnh đại diện cho agent
│   │       ├── start.png           # Hình ảnh điểm bắt đầu
│   │       └── goal.png            # Hình ảnh điểm đích
│   │
│   ├── templates/
│   │   ├── index.html              # Trang chính
│   │   ├── simulation.html         # Trang mô phỏng
│   │   └── analysis.html           # Trang phân tích kết quả
│   │
│   └── models/                     # Thư mục chứa các mô hình được xuất để sử dụng trong web
│       └── exported/               # Các mô hình được chuyển đổi để sử dụng trên web
│           ├── q_learning_10x10.json  # Q-table được chuyển thành JSON
│           └── sarsa_10x10.json       # Q-table được chuyển thành JSON
│
├── scripts/
│   ├── convert_model.py            # Script chuyển đổi từ pkl sang JSON
│   ├── generate_dataset.py         # Script để tạo bộ mê cung đa dạng
│   └── export_results.py           # Script xuất kết quả phân tích
│
├── README.md                       # Tài liệu hướng dẫn
├── requirements.txt                # Các thư viện Python cần thiết
└── server.py                       # Script chính để chạy ứng dụng web