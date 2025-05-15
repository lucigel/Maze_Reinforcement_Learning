maze-solver-project/
├── backend/                           # Backend - Python FastAPI
│   ├── maze_generators/               # Thuật toán sinh mê cung (vẫn giữ nhưng chỉ để training)
│   │   ├── __init__.py
│   │   ├── base_generator.py          
│   │   ├── dfs_generator.py           
│   │   ├── prim_generator.py          
│   │   └── wilson_generator.py        
│   │
│   ├── rl_agents/                     # Thuật toán học tăng cường
│   │   ├── __init__.py
│   │   ├── base_agent.py              
│   │   ├── q_learning.py              
│   │   └── sarsa.py                   
│   │
│   ├── environment/                   # Môi trường mê cung
│   │   ├── __init__.py
│   │   └── maze_env.py                # Cần bổ sung khả năng nhận mê cung từ frontend
│   │
│   ├── utils/                         
│   │   ├── __init__.py
│   │   ├── config.py                  
│   │   ├── data_handler.py            
│   │   └── visualization.py           
│   │
│   ├── models/                        # Lưu mô hình đã huấn luyện
│   │   ├── q_learning/
│   │   │   ├── maze_10x10.pkl         
│   │   │   └── maze_20x20.pkl         
│   │   └── sarsa/
│   │       ├── maze_10x10.pkl
│   │       └── maze_20x20.pkl
│   │
│   ├── api.py                         # Thêm endpoint solve_maze
│   └── [các file khác]
│
├── frontend/                          # Frontend - React
│   ├── public/
│   │   └── [các file hiện có]
│   │
│   ├── src/
│   │   ├── App.js                     
│   │   ├── index.js                   
│   │   │
│   │   ├── api/
│   │   │   └── mazeApi.js             # Thêm hàm solveMaze để gọi API
│   │   │
│   │   ├── algorithms/                # Thư mục mới: Thuật toán JS cho frontend
│   │   │   ├── mazeGenerators/        
│   │   │   │   ├── dfsGenerator.js    # Thuật toán DFS bằng JS
│   │   │   │   ├── primGenerator.js   # Thuật toán Prim bằng JS
│   │   │   │   └── wilsonGenerator.js # Thuật toán Wilson bằng JS
│   │   │   └── utils/
│   │   │       ├── mazeUtils.js       # Các tiện ích xử lý mê cung
│   │   │       └── animationUtils.js  # Xử lý animation
│   │   │
│   │   ├── components/
│   │   │   ├── Common/                # Components chung
│   │   │   │   └── [các file hiện có]
│   │   │   │
│   │   │   ├── MazeGenerator/         # Components cho trang sinh mê cung
│   │   │   │   ├── GeneratorPage.js   # Gọi thuật toán JS thay vì API
│   │   │   │   ├── GeneratorControls.js
│   │   │   │   ├── MazeCanvas.js      # Canvas hiển thị quá trình sinh
│   │   │   │   ├── StepControls.js    
│   │   │   │   └── SaveMazeButton.js  # Component mới: nút Lưu mê cung
│   │   │   │
│   │   │   └── MazeSolver/            # Components cho trang giải mê cung
│   │   │       ├── SolverPage.js
│   │   │       ├── ModelSelector.js   
│   │   │       ├── MazeSelector.js    # Component mới: chọn mê cung đã lưu
│   │   │       ├── MazeVisualization.js
│   │   │       ├── MouseVisualization.js
│   │   │       ├── ValueFunctionHeatmap.js # Hiển thị Q-values từ backend
│   │   │       └── PerformanceStats.js
│   │   │
│   │   ├── hooks/                     # Custom hooks
│   │   │   ├── useMazeGenerator.js    # Hook để quản lý quá trình sinh mê cung
│   │   │   └── useMazeSolver.js       # Hook để quản lý quá trình giải mê cung
│   │   │
│   │   ├── context/                   # Thư mục mới: Context API
│   │   │   └── MazeContext.js         # Lưu trữ trạng thái mê cung toàn cục
│   │   │
│   │   ├── utils/                     # Tiện ích chung
│   │   │   ├── storageUtils.js        # Xử lý localStorage/sessionStorage
│   │   │   └── mazeConversion.js      # Chuyển đổi giữa các định dạng mê cung
│   │   │
│   │   └── styles/                    
│   │       └── [các file hiện có]
│   │
│   └── [các file khác]
└── [các file khác]

### Scripts training 

```python
    python training.py --agent sarsa --maze dfs --size small --episodes 10000 --lr 0.2 --gamma 0.99 --epsilon 1.0 --decay 0.999
```