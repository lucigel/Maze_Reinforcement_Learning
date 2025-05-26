# get_heatmaps.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from rl_agents.q_learning import QLearningAgent
from rl_agents.sarsa import SARSAAgent
from rl_agents.dqn_agent import DQNAgent
from enviroment.maze_env import MazeEnvironment
from maze_generators.dfs_generator import DFSMazeGenerator
from utils.data_handler import load_model


def create_heatmaps(model_type='q_learning', maze_size=11, episodes=2000, output_dir='results/heatmaps'):
    """
    Tạo heatmap chính sách và hành vi cho agent
    
    Args:
        model_type: Loại model ('q_learning', 'sarsa', 'dqn')
        maze_size: Kích thước mê cung
        episodes: Số episode đã train (để tìm đúng file model)
        output_dir: Thư mục lưu kết quả
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
    elif model_type == 'dqn':
        # Tạo DQN agent với device phù hợp
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            device=device
        )
    else:
        raise ValueError(f"Loại mô hình không hợp lệ: {model_type}")
    
    # Xác định đường dẫn model
    if model_type == 'dqn':
        # DQN sử dụng file .pth
        model_filename = f"dqn_{maze_size}x{maze_size}_{episodes}ep.pth"
        model_path = f"models/dqn/{model_filename}"
        
        # Load model cho DQN
        try:
            agent.load_model(model_path)
            # Cần set maze cho DQN để xử lý state
            agent.current_maze = env.maze
            print(f"Đã tải mô hình DQN từ {model_path}")
        except Exception as e:
            print(f"Lỗi khi tải mô hình DQN: {e}")
            return
    else:
        # Q-Learning và SARSA sử dụng file .pkl
        model_filename = f"{model_type}_{maze_size}x{maze_size}_{episodes}ep.pkl"
        model_path = f"models/{model_type}/{model_filename}"
        
        try:
            load_model(agent, model_path)
            print(f"Đã tải mô hình từ {model_path}")
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return
    
    # 1. Tạo heatmap chính sách (Policy heatmap)
    policy = agent.get_policy()
    
    plt.figure(figsize=(12, 10))
    
    # Hiển thị mê cung làm nền
    maze_display = np.where(env.maze == 1, -1, np.nan)  # Tường = -1, đường = NaN
    plt.imshow(maze_display, cmap='gray', alpha=0.3)
    
    # Overlay policy
    masked_policy = np.ma.masked_where(env.maze == 1, policy)
    im = plt.imshow(masked_policy, cmap='viridis', alpha=0.8)
    
    # Thêm mũi tên chỉ hướng
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Lên, xuống, trái, phải
    arrow_symbols = ['↑', '↓', '←', '→']
    
    for r in range(maze_size):
        for c in range(maze_size):
            if env.maze[r, c] == 0:  # Chỉ hiển thị ở các ô đường đi
                action = policy[r, c]
                plt.text(c, r, arrow_symbols[action], 
                        ha='center', va='center', 
                        color='white', fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.set_label('Hành động', fontsize=12)
    cbar.set_ticklabels(['Lên', 'Xuống', 'Trái', 'Phải'])
    
    plt.title(f'Policy Heatmap - {model_type.upper()}', fontsize=16)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_type}_policy_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Tạo heatmap giá trị (Value heatmap)
    value_function = agent.get_value_function()
    
    plt.figure(figsize=(12, 10))
    
    # Hiển thị mê cung làm nền
    plt.imshow(maze_display, cmap='gray', alpha=0.3)
    
    # Overlay values
    masked_values = np.ma.masked_where(env.maze == 1, value_function)
    im = plt.imshow(masked_values, cmap='hot', alpha=0.8)
    
    # Thêm giá trị số
    for r in range(maze_size):
        for c in range(maze_size):
            if env.maze[r, c] == 0:
                value = value_function[r, c]
                color = 'white' if value < np.mean(value_function) else 'black'
                plt.text(c, r, f'{value:.1f}', 
                        ha='center', va='center', 
                        color=color, fontsize=8)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Giá trị', fontsize=12)
    
    plt.title(f'Value Heatmap - {model_type.upper()}', fontsize=16)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_type}_value_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Tạo heatmap lượt truy cập (Visitation heatmap)
    # Reset state visits để đếm lại
    agent.state_visits = np.zeros(state_size)
    
    # Đánh giá agent để thu thập dữ liệu lượt thăm
    original_epsilon = agent.epsilon
    agent.epsilon = 0.1  # Một chút khám phá để heatmap đa dạng hơn
    
    num_eval_episodes = 100
    for episode in range(num_eval_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 1000:
            if model_type == 'dqn':
                action = agent.choose_action(state, env.maze)
            else:
                action = agent.choose_action(state)
                
            next_state, reward, done, _ = env.step(action)
            state = next_state
            steps += 1
    
    agent.epsilon = original_epsilon
    
    plt.figure(figsize=(12, 10))
    
    # Hiển thị mê cung làm nền
    plt.imshow(maze_display, cmap='gray', alpha=0.3)
    
    # Overlay visitation counts
    masked_visits = np.ma.masked_where(env.maze == 1, agent.state_visits)
    im = plt.imshow(masked_visits, cmap='Blues', alpha=0.8)
    
    # Thêm số lượt thăm
    for r in range(maze_size):
        for c in range(maze_size):
            if env.maze[r, c] == 0 and agent.state_visits[r, c] > 0:
                visits = int(agent.state_visits[r, c])
                plt.text(c, r, str(visits), 
                        ha='center', va='center', 
                        color='black', fontsize=8)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Số lần thăm', fontsize=12)
    
    plt.title(f'Visitation Heatmap - {model_type.upper()} ({num_eval_episodes} episodes)', fontsize=16)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_type}_visitation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Tạo combined heatmap (3 heatmap cạnh nhau)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Policy
    masked_policy = np.ma.masked_where(env.maze == 1, policy)
    im1 = axes[0].imshow(masked_policy, cmap='viridis')
    axes[0].set_title(f'Policy - {model_type.upper()}', fontsize=14)
    plt.colorbar(im1, ax=axes[0], ticks=[0, 1, 2, 3], 
                label='Action', 
                orientation='horizontal',
                pad=0.1).set_ticklabels(['Up', 'Down', 'Left', 'Right'])
    
    # Value
    masked_values = np.ma.masked_where(env.maze == 1, value_function)
    im2 = axes[1].imshow(masked_values, cmap='hot')
    axes[1].set_title(f'Value Function - {model_type.upper()}', fontsize=14)
    plt.colorbar(im2, ax=axes[1], label='Value', orientation='horizontal', pad=0.1)
    
    # Visitation
    masked_visits = np.ma.masked_where(env.maze == 1, agent.state_visits)
    im3 = axes[2].imshow(masked_visits, cmap='Blues')
    axes[2].set_title(f'State Visits - {model_type.upper()}', fontsize=14)
    plt.colorbar(im3, ax=axes[2], label='Visits', orientation='horizontal', pad=0.1)
    
    # Remove ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_type}_combined_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Đã tạo heatmap thành công và lưu vào {output_dir}")


if __name__ == "__main__":
    # Cấu hình
    maze_size = 15  # Kích thước mê cung (15x15)
    episodes = 2000  # Số episode đã train
    
    # Tạo heatmap cho Q-Learning
    # create_heatmaps(
    #     model_type='q_learning', 
    #     maze_size=maze_size, 
    #     episodes=episodes,
    #     output_dir='results/q_learning_heatmaps'
    # )
    
    # # Tạo heatmap cho SARSA
    # create_heatmaps(
    #     model_type='sarsa', 
    #     maze_size=maze_size,
    #     episodes=episodes, 
    #     output_dir='results/sarsa_heatmaps'
    # )
    
    # Tạo heatmap cho DQN
    create_heatmaps(
        model_type='dqn', 
        maze_size=maze_size,
        episodes=episodes,
        output_dir='results/dqn_heatmaps'
    )
    
    print("\nĐã tạo xong tất cả heatmaps!")