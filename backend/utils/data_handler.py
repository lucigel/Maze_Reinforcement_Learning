# data_handler.py
import os
import pickle
import numpy as np
import json
from typing import Dict, Any, Optional, Tuple, List

def save_model(agent, model_path: str) -> None:
    """
    Lưu mô hình học tăng cường.
    
    Args:
        agent: Agent học tăng cường (Q-Learning hoặc SARSA)
        model_path: Đường dẫn để lưu mô hình
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Lưu mô hình
    agent.save_model(model_path)
    print(f"Đã lưu mô hình tại: {model_path}")

def load_model(agent, model_path: str) -> Any:
    """
    Tải mô hình học tăng cường.
    
    Args:
        agent: Agent học tăng cường (Q-Learning hoặc SARSA)
        model_path: Đường dẫn để tải mô hình
        
    Returns:
        Agent đã được tải mô hình
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình tại: {model_path}")
    
    agent.load_model(model_path)
    print(f"Đã tải mô hình từ: {model_path}")
    return agent

def save_training_history(history: Dict[str, List], save_path: str) -> None:
    """
    Lưu lịch sử huấn luyện.
    
    Args:
        history: Dictionary chứa dữ liệu lịch sử (reward, steps)
        save_path: Đường dẫn để lưu lịch sử
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **history)
    print(f"Đã lưu lịch sử huấn luyện tại: {save_path}")

def load_training_history(load_path: str) -> Dict[str, np.ndarray]:
    """
    Tải lịch sử huấn luyện.
    
    Args:
        load_path: Đường dẫn để tải lịch sử
        
    Returns:
        Dictionary chứa dữ liệu lịch sử
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Không tìm thấy lịch sử tại: {load_path}")
    
    data = np.load(load_path)
    return {key: data[key] for key in data.files}

def export_model_to_json(agent, json_path: str) -> None:
    """
    Xuất mô hình sang định dạng JSON để sử dụng trên web.
    
    Args:
        agent: Agent học tăng cường (Q-Learning hoặc SARSA)
        json_path: Đường dẫn để lưu file JSON
    """
    agent.export_to_json(json_path)
    print(f"Đã xuất mô hình sang JSON tại: {json_path}")

def compare_models(model1_path: str, model2_path: str, metric: str = "steps") -> Tuple[float, float]:
    """
    So sánh hiệu suất của hai mô hình dựa trên lịch sử huấn luyện.
    
    Args:
        model1_path: Đường dẫn đến lịch sử huấn luyện của mô hình 1
        model2_path: Đường dẫn đến lịch sử huấn luyện của mô hình 2
        metric: Tiêu chí so sánh ("steps" hoặc "rewards")
        
    Returns:
        Tuple chứa giá trị trung bình của tiêu chí cho mỗi mô hình
    """
    history1 = load_training_history(model1_path)
    history2 = load_training_history(model2_path)
    
    if metric == "steps":
        model1_metric = np.mean(history1["steps"][-100:])  # Lấy 100 episode cuối
        model2_metric = np.mean(history2["steps"][-100:])
    elif metric == "rewards":
        model1_metric = np.mean(history1["rewards"][-100:])
        model2_metric = np.mean(history2["rewards"][-100:])
    else:
        raise ValueError(f"Tiêu chí không hợp lệ: {metric}")
    
    return model1_metric, model2_metric

def prepare_visualization_data(agent, maze, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Chuẩn bị dữ liệu để hiển thị trực quan.
    
    Args:
        agent: Agent học tăng cường
        maze: Ma trận mê cung
        output_path: Đường dẫn để lưu dữ liệu (tùy chọn)
        
    Returns:
        Dictionary chứa dữ liệu để hiển thị
    """
    height, width = maze.shape
    
    # Lấy chính sách và hàm giá trị
    policy = agent.get_policy()
    value_function = agent.get_value_function()
    
    # Chuẩn bị dữ liệu
    visualization_data = {
        "maze": maze.tolist(),
        "policy": policy.tolist(),
        "value_function": value_function.tolist(),
        "state_visits": agent.state_visits.tolist(),
        "q_table": [[agent.q_table[r, c, :].tolist() for c in range(width)] for r in range(height)]
    }
    
    # Lưu dữ liệu nếu được yêu cầu
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(visualization_data, f)
        print(f"Đã lưu dữ liệu hiển thị tại: {output_path}")
    
    return visualization_data

def analyze_training_convergence(history: Dict[str, List], window_size: int = 10) -> Dict[str, np.ndarray]:
    """
    Phân tích sự hội tụ trong quá trình huấn luyện.
    
    Args:
        history: Dictionary chứa lịch sử huấn luyện
        window_size: Kích thước cửa sổ cho việc làm mịn
        
    Returns:
        Dictionary chứa dữ liệu phân tích
    """
    rewards = np.array(history["rewards"])
    steps = np.array(history["steps"])
    
    # Tính trung bình di động
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    smoothed_steps = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
    
    # Tính độ lệch chuẩn
    reward_std = np.std(rewards[-100:])
    steps_std = np.std(steps[-100:])
    
    # Tính tốc độ hội tụ (dựa trên độ dốc)
    episodes = np.arange(len(rewards))
    reward_slope = np.polyfit(episodes[-100:], rewards[-100:], 1)[0]
    steps_slope = np.polyfit(episodes[-100:], steps[-100:], 1)[0]
    
    return {
        "smoothed_rewards": smoothed_rewards,
        "smoothed_steps": smoothed_steps,
        "reward_std": reward_std,
        "steps_std": steps_std,
        "reward_slope": reward_slope,
        "steps_slope": steps_slope
    }