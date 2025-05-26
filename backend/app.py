from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import os
import pickle
import torch
from enviroment.maze_env import MazeEnvironment
from rl_agents.q_learning import QLearningAgent
from rl_agents.sarsa import SARSAAgent
from rl_agents.dqn_agent import DQNAgent  # Import DQN Agent
from maze_generators.dfs_generator import DFSMazeGenerator
from maze_generators.prim_generator import PrimMazeGenerator
from maze_generators.wilson_generator import WilsonMazeGenerator

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


PATH = 0
WALL = 1
START = 2
GOAL = 3

# Models
class MazeRequest(BaseModel):
    size: int
    generator: str

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
    height, width = maze_size
    
    # Chọn mô hình phù hợp với kích thước mê cung
    if height <= 11:
        size_str = "11x11"
    elif height <= 15:
        size_str = "15x15"
    else:
        size_str = "21x21"
    
    print(f"Đang tải mô hình {algorithm} cho kích thước {size_str}")
    
    # Xây dựng đường dẫn file
    if algorithm == "q_learning":
        file_path = os.path.join(MODEL_DIR, "q_learning", f"maze_{size_str}_2000ep.pkl")
        if not os.path.exists(file_path):
            print(f"Không tìm thấy: {file_path}")
            file_path = os.path.join(MODEL_DIR, "q_learning", "q_learning_11x11_2000ep.pkl")
    elif algorithm == "sarsa":
        file_path = os.path.join(MODEL_DIR, "sarsa", f"sarsa_{size_str}_2000ep.pkl")
        if not os.path.exists(file_path):
            print(f"Không tìm thấy: {file_path}")
            file_path = os.path.join(MODEL_DIR, "sarsa", "sarsa_11x11_2000ep.pkl")
    elif algorithm == "dqn":
        # Xử lý cho DQN, ưu tiên file .pth, nếu không có thì tìm file .pkl
        file_path = os.path.join(MODEL_DIR, "dqn", f"dqn_{size_str}_2000ep.pkl")
        if not os.path.exists(file_path):
            print(f"Không tìm thấy: {file_path}")
            # Tìm các tệp tin .pth khác
            dqn_dir = os.path.join(MODEL_DIR, "dqn")
            if os.path.exists(dqn_dir):
                pth_files = [f for f in os.listdir(dqn_dir) if f.endswith(".pth")]
                if pth_files:
                    file_path = os.path.join(dqn_dir, pth_files[0])
                    print(f"Sử dụng tệp thay thế: {file_path}")
                else:
                    # Nếu không có tệp .pth nào, tìm tệp .pkl
                    pkl_files = [f for f in os.listdir(dqn_dir) if f.endswith(".pkl")]
                    if pkl_files:
                        file_path = os.path.join(dqn_dir, pkl_files[0])
                        print(f"Sử dụng tệp thay thế: {file_path}")
                    else:
                        file_path = None
            else:
                file_path = None
    else:
        raise ValueError(f"Thuật toán không được hỗ trợ: {algorithm}")
    
    # Kiểm tra xem file có tồn tại không
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file mô hình cho {algorithm} kích thước {size_str}")
    
    print(f"Đang tải mô hình từ: {file_path}")
    
    # Tạo agent và load dữ liệu dựa vào thuật toán
    if algorithm == "q_learning":
        agent = QLearningAgent(state_size=maze_size, action_size=4)
        # Tải mô hình
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load dữ liệu vào agent
        agent.q_table = model_data['q_table']
        agent.state_size = model_data['state_size']
        agent.action_size = model_data['action_size']
        agent.epsilon = 0.0  # Đặt epsilon = 0 để agent không khám phá ngẫu nhiên
        agent.lr = model_data.get('lr', 0.1)
        agent.gamma = model_data.get('gamma', 0.99)
        
        if 'state_visits' in model_data:
            agent.state_visits = model_data['state_visits']
    
    elif algorithm == "sarsa":
        agent = SARSAAgent(state_size=maze_size, action_size=4)
        # Tải mô hình
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load dữ liệu vào agent
        agent.q_table = model_data['q_table']
        agent.state_size = model_data['state_size']
        agent.action_size = model_data['action_size']
        agent.epsilon = 0.0  # Đặt epsilon = 0 để agent không khám phá ngẫu nhiên
        agent.lr = model_data.get('lr', 0.1)
        agent.gamma = model_data.get('gamma', 0.99)
        
        if 'state_visits' in model_data:
            agent.state_visits = model_data['state_visits']
    
    elif algorithm == "dqn":
        # Khởi tạo DQN agent
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = DQNAgent(state_size=maze_size, action_size=4, device=device)
        
        # Kiểm tra xem file là .pth hay .pkl
        if file_path.endswith('.pth'):
            # Tải mô hình PyTorch
            try:
                agent.load_model(file_path)
                print("Đã tải mô hình PyTorch thành công")
            except Exception as e:
                print(f"Lỗi khi tải mô hình PyTorch: {e}")
                # Nếu có lỗi, thử tải file .pkl tương ứng
                pkl_path = file_path.replace('.pth', '.pkl')
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Cập nhật thuộc tính cơ bản
                    agent.state_size = model_data['state_size']
                    agent.action_size = model_data['action_size']
                    agent.epsilon = 0.0
                    
                    if 'state_visits' in model_data:
                        agent.state_visits = model_data['state_visits']
        else:
            # Tải thông tin từ file .pkl
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Cập nhật thuộc tính cơ bản
            agent.state_size = model_data['state_size']
            agent.action_size = model_data['action_size']
            agent.epsilon = 0.0
            
            if 'state_visits' in model_data:
                agent.state_visits = model_data['state_visits']
    
    print(f"Đã tải mô hình {algorithm} thành công")
    return agent

@app.get("/", response_class=HTMLResponse)
async def root():
    """Trả về trang chủ (HTML)"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/api/generate-maze")
async def generate_maze(request: MazeRequest):
    """Tạo mê cung mới với kích thước và thuật toán được chỉ định"""
    size = request.size
    generator_type = request.generator
    
    # Tạo mê cung với bộ sinh phù hợp
    if generator_type == "dfs":
        generator = DFSMazeGenerator(width=size, height=size)
    elif generator_type == "prim":
        generator = PrimMazeGenerator(width=size, height=size)
    elif generator_type == "wilson":
        generator = WilsonMazeGenerator(width=size, height=size)
    else:
        raise HTTPException(status_code=400, detail="Thuật toán sinh mê cung không hỗ trợ")
    
    maze = generator.generate()
    start_pos = generator.start_pos
    goal_pos = generator.goal_pos
    
    return {
        "maze": maze.tolist(),
        "start_pos": start_pos,
        "goal_pos": goal_pos
    }

@app.post("/api/solve-maze")
async def solve_maze(data: MazeData) -> SolveResponse:
    """Giải mê cung với agent đã huấn luyện và trả về đường đi."""
    try:
        # Chuyển đổi maze từ danh sách 2D thành mảng numpy
        maze_array = np.array(data.maze)
        height, width = maze_array.shape
        
        print(f"Kích thước mê cung: {height}x{width}")
        
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
        
        print(f"Vị trí bắt đầu: {start_pos}, Vị trí đích: {goal_pos}")
        
        # Tạo môi trường mê cung
        env = MazeEnvironment(maze=maze_array, start_pos=start_pos, goal_pos=goal_pos)
        
        # Tải agent phù hợp
        algorithm = data.algorithm.lower()
        if algorithm not in ["q_learning", "sarsa", "dqn"]:
            raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}")
            
        agent = load_agent(algorithm, (height, width))
        
        # Mô phỏng đường đi của agent
        path = []
        total_reward = 0
        
        state = env.reset()
        path.append(state)
        done = False
        steps = 0
        
        # Giới hạn số bước tối đa để tránh chạy quá lâu
        max_steps = min(1000, height * width * 5)  # Giới hạn hợp lý dựa vào kích thước mê cung
        
        print(f"Bắt đầu giải mê cung từ vị trí {state} với tối đa {max_steps} bước")
        
        while not done and steps < max_steps:
            # Chọn hành động tối ưu từ agent
            if algorithm == "dqn":
                # Đối với DQN, cần truyền cả maze cho agent
                action = agent.choose_action(state, env.maze)
            else:
                # Đối với Q-learning và SARSA, chỉ cần truyền state
                action = np.argmax(agent.q_table[state[0], state[1]])
            
            # Thực hiện hành động
            next_state, reward, done, _ = env.step(action)
            
            if steps % 10 == 0:  # Chỉ in log mỗi 10 bước để giảm lượng log
                print(f"Bước {steps}: Từ {state} thực hiện hành động {action} đến {next_state}, reward: {reward}, done: {done}")
            
            # Cập nhật biến theo dõi
            state = next_state
            path.append(state)
            total_reward += reward
            steps += 1
            
            # Thoát nếu đã đến đích
            if done:
                print(f"Đã đến đích sau {steps} bước!")
                break
        
        # Nếu đã đạt đến số bước tối đa nhưng chưa đến đích
        if steps >= max_steps and not done:
            print(f"Đã đạt đến số bước tối đa ({max_steps}) mà chưa đến đích")
        
        # Lấy ma trận giá trị Q hoặc value function
        q_values = {}
        if algorithm in ["q_learning", "sarsa"]:
            q_values = {
                "table": agent.q_table.tolist(),
                "value_function": agent.get_value_function().tolist(),
                "policy": agent.get_policy().tolist()
            }
        elif algorithm == "dqn":
            # Đặt maze hiện tại cho DQN agent để lấy value function và policy
            agent.current_maze = env.maze
            q_values = {
                "table": [],  # DQN không có q_table truyền thống
                "value_function": agent.get_value_function().tolist(),
                "policy": agent.get_policy().tolist()
            }
        
        # Trả về kết quả
        print(f"Kết quả giải: {steps} bước, tổng reward: {total_reward}, đã đến đích: {done}")
        return SolveResponse(
            path=path,
            q_values=q_values,
            steps=steps,
            total_reward=float(total_reward)
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models-info")
async def get_models_info():
    """Trả về thông tin về các mô hình đã huấn luyện."""
    try:
        models_info = {
            "q_learning": [],
            "sarsa": [],
            "dqn": []  # Thêm DQN vào danh sách
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
        
        # Lấy thông tin mô hình DQN
        dqn_dir = os.path.join(MODEL_DIR, "dqn")
        if os.path.exists(dqn_dir):
            for file in os.listdir(dqn_dir):
                if file.endswith(".pth") or file.endswith(".pkl"):
                    models_info["dqn"].append(file)
        
        return models_info
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Thêm để phục vụ HTML
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_index(full_path: str):
    """Serve index.html for all routes for SPA behavior"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Thêm endpoint để kiểm tra sức khỏe API
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)