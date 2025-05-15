from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import os
import pickle
from enviroment.maze_env import MazeEnvironment
from rl_agents.q_learning import QLearningAgent
from rl_agents.sarsa import SARSAAgent

app = FastAPI()

# Cấu hình CORS để frontend có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các origin trong môi trường phát triển
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Các hằng số cho mê cung
PATH = 0
WALL = 1
START = 2
GOAL = 3

# Models cho Pydantic
class MazeData(BaseModel):
    maze: List[List[int]]
    algorithm: str
    start_pos: Optional[Tuple[int, int]] = None
    goal_pos: Optional[Tuple[int, int]] = None

class SolveResponse(BaseModel):
    path: List[Tuple[int, int]]
    q_values: Dict[str, Any]
    steps: int
    total_reward: float

# Đường dẫn tới các mô hình đã huấn luyện
MODEL_DIR = "models"

def load_agent(algorithm: str, maze_size: Tuple[int, int]) -> Any:
    """Tải agent đã huấn luyện từ file."""
    # Xác định đường dẫn file dựa trên thuật toán và kích thước mê cung
    height, width = maze_size
    
    # Tìm mô hình phù hợp nhất với kích thước mê cung
    if height <= 10 and width <= 10:
        size_str = "10x10"
    elif height <= 15 and width <= 15:
        size_str = "15x15"
    else:
        size_str = "20x20"
    
    # Xây dựng đường dẫn file
    if algorithm == "q_learning":
        file_path = os.path.join(MODEL_DIR, "q_learning", f"maze_{size_str}.pkl")
        if not os.path.exists(file_path):
            file_path = os.path.join(MODEL_DIR, "q_learning", "q_learning_final.pkl")
    else:  # SARSA
        file_path = os.path.join(MODEL_DIR, "sarsa", f"maze_{size_str}.pkl")
        if not os.path.exists(file_path):
            file_path = os.path.join(MODEL_DIR, "sarsa", "sarsa_final.pkl")
    
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file mô hình cho {algorithm} kích thước {size_str}")
    
    # Tải mô hình
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Tạo agent và load dữ liệu
    if algorithm == "q_learning":
        agent = QLearningAgent(state_size=maze_size, action_size=4)
    else:
        agent = SARSAAgent(state_size=maze_size, action_size=4)
    
    # Load dữ liệu vào agent
    agent.q_table = model_data['q_table']
    agent.state_size = model_data['state_size']
    agent.action_size = model_data['action_size']
    agent.epsilon = model_data['epsilon']
    agent.lr = model_data['lr']
    agent.gamma = model_data['gamma']
    
    if 'state_visits' in model_data:
        agent.state_visits = model_data['state_visits']
    
    return agent

@app.post("/api/solve-maze")
async def solve_maze(data: MazeData) -> SolveResponse:
    """Giải mê cung với agent đã huấn luyện và trả về đường đi."""
    try:
        # Chuyển đổi maze từ danh sách 2D thành mảng numpy
        maze_array = np.array(data.maze)
        height, width = maze_array.shape
        
        # Tìm vị trí bắt đầu và đích nếu không được chỉ định
        start_pos = data.start_pos
        goal_pos = data.goal_pos
        
        if start_pos is None:
            start_positions = np.where(maze_array == START)
            if len(start_positions[0]) > 0:
                start_pos = (int(start_positions[0][0]), int(start_positions[1][0]))
            else:
                # Tìm vị trí đường đi đầu tiên nếu không có START
                path_positions = np.where(maze_array == PATH)
                if len(path_positions[0]) > 0:
                    start_pos = (int(path_positions[0][0]), int(path_positions[1][0]))
                else:
                    raise ValueError("Không tìm thấy vị trí bắt đầu trong mê cung")
        
        if goal_pos is None:
            goal_positions = np.where(maze_array == GOAL)
            if len(goal_positions[0]) > 0:
                goal_pos = (int(goal_positions[0][0]), int(goal_positions[1][0]))
            else:
                # Tìm vị trí đường đi cuối cùng nếu không có GOAL
                path_positions = np.where(maze_array == PATH)
                if len(path_positions[0]) > 0:
                    goal_pos = (int(path_positions[0][-1]), int(path_positions[1][-1]))
                else:
                    raise ValueError("Không tìm thấy vị trí đích trong mê cung")
        
        # Tạo môi trường mê cung
        env = MazeEnvironment(maze=maze_array, start_pos=start_pos, goal_pos=goal_pos)
        
        # Tải agent phù hợp
        agent = load_agent(data.algorithm, (height, width))
        
        # Mô phỏng đường đi của agent
        path = []
        total_reward = 0
        
        state = env.reset()
        path.append(state)
        done = False
        steps = 0
        
        while not done and steps < 1000:  # Giới hạn tối đa 1000 bước
            # Chọn hành động tối ưu từ Q-table
            action = np.argmax(agent.q_table[state[0], state[1]])
            
            # Thực hiện hành động
            next_state, reward, done, _ = env.step(action)
            
            # Cập nhật biến theo dõi
            state = next_state
            path.append(state)
            total_reward += reward
            steps += 1
            
            # Thoát nếu đã đến đích hoặc quá số bước tối đa
            if done or steps >= 1000:
                break
        
        # Lấy ma trận giá trị Q
        q_values = {
            "table": agent.q_table.tolist(),
            "value_function": agent.get_value_function().tolist(),
            "policy": agent.get_policy().tolist()
        }
        
        # Trả về kết quả
        return SolveResponse(
            path=path,
            q_values=q_values,
            steps=steps,
            total_reward=float(total_reward)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models-info")
async def get_models_info():
    """Trả về thông tin về các mô hình đã huấn luyện."""
    try:
        models_info = {
            "q_learning": [],
            "sarsa": []
        }
        
        # Lấy thông tin mô hình Q-Learning
        q_learning_dir = os.path.join(MODEL_DIR, "q_learning")
        if os.path.exists(q_learning_dir):
            for file in os.listdir(q_learning_dir):
                if file.endswith(".pkl"):
                    models_info["q_learning"].append(file)
        
        # Lấy thông tin mô hình SARSA
        sarsa_dir = os.path.join(MODEL_DIR, "sarsa")
        if os.path.exists(sarsa_dir):
            for file in os.listdir(sarsa_dir):
                if file.endswith(".pkl"):
                    models_info["sarsa"].append(file)
        
        return models_info
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Thêm endpoint để kiểm tra sức khỏe API
@app.get("/health")
async def health_check():
    return {"status": "ok"}