# get_heatmaps.py
import os
import numpy as np
import matplotlib.pyplot as plt
from rl_agents.q_learning import QLearningAgent
from rl_agents.sarsa import SARSAAgent
from enviroment.maze_env import MazeEnvironment
from maze_generators.dfs_generator import DFSMazeGenerator
from utils.data_handler import load_model

def create_heatmaps(model_type='q_learning', maze_size=11, output_dir='results/heatmaps'):
    """
    Tạo heatmap chính sách và hành vi cho agent
    """
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo môi trường mê cung
    maze_generator = DFSMazeGenerator(maze_size, maze_size)
    env = MazeEnvironment(maze_generator=maze_generator)
    
    # Tạo agent
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    
    if model_type == 'q_learning':
        agent = QLearningAgent(state_size, action_size)
    elif model_type == 'sarsa':
        agent = SARSAAgent(state_size, action_size)
    else:
        raise ValueError(f"Loại mô hình không hợp lệ: {model_type}")
    
    # Tải mô hình
    model_path = f"models/{model_type}/{model_type}_final.pkl"
    try:
        load_model(agent, model_path)
        print(f"Đã tải mô hình từ {model_path}")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return
    
    # 1. Tạo heatmap chính sách (Policy heatmap)
    policy = agent.get_policy()
    
    plt.figure(figsize=(10, 10))
    im = plt.imshow(policy, cmap='viridis')
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.set_label('Hành động')
    cbar.set_ticklabels(['Lên', 'Xuống', 'Trái', 'Phải'])
    plt.title(f'Policy Heatmap - {model_type}')
    plt.savefig(os.path.join(output_dir, f'{model_type}_policy_heatmap.png'), dpi=300)
    plt.close()
    
    # 2. Tạo heatmap giá trị (Value heatmap)
    value_function = agent.get_value_function()
    
    plt.figure(figsize=(10, 10))
    im = plt.imshow(value_function, cmap='hot')
    cbar = plt.colorbar(im)
    cbar.set_label('Giá trị')
    plt.title(f'Value Heatmap - {model_type}')
    plt.savefig(os.path.join(output_dir, f'{model_type}_value_heatmap.png'), dpi=300)
    plt.close()
    
    # 3. Tạo heatmap lượt truy cập (Visitation heatmap)
    # Đánh giá agent để thu thập dữ liệu lượt thăm
    # Chạy một số episode để cập nhật state_visits
    original_epsilon = agent.epsilon
    agent.epsilon = 0.1  # Một chút khám phá để heatmap đa dạng hơn
    
    for episode in range(100):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 1000:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            steps += 1
    
    agent.epsilon = original_epsilon
    
    plt.figure(figsize=(10, 10))
    im = plt.imshow(agent.state_visits, cmap='Blues')
    cbar = plt.colorbar(im)
    cbar.set_label('Số lần thăm')
    plt.title(f'Visitation Heatmap - {model_type}')
    plt.savefig(os.path.join(output_dir, f'{model_type}_visitation_heatmap.png'), dpi=300)
    plt.close()
    
    print(f"Đã tạo heatmap thành công và lưu vào {output_dir}")

if __name__ == "__main__":
    # Tạo heatmap cho Q-Learning
    create_heatmaps(model_type='q_learning', maze_size=10, output_dir='results/q_learning_heatmaps')
    
    # Tạo heatmap cho SARSA
    create_heatmaps(model_type='sarsa', maze_size=10, output_dir='results/sarsa_heatmaps')