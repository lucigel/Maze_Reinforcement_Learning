# visualization.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import seaborn as sns 
from typing import Dict, List, Optional, Tuple, Any
import os
import pandas as pd

# Định nghĩa màu sắc cho biểu đồ
COLORS = {
    'wall': '#434343',      # Tường (xám đậm)
    'path': '#FFFFFF',      # Đường đi (trắng)
    'start': '#4CAF50',     # Điểm bắt đầu (xanh lá)
    'goal': '#F44336',      # Điểm đích (đỏ)
    'agent': '#2196F3',     # Agent (xanh dương)
    'value_low': '#FFF9C4', # Giá trị thấp (vàng nhạt)
    'value_high': '#FFC107',# Giá trị cao (vàng đậm)
    'visit_low': '#E1F5FE', # Thăm ít (xanh nhạt)
    'visit_high': '#0288D1', # Thăm nhiều (xanh đậm)
    'q_learning': '#4CAF50',# Q-Learning (xanh lá)
    'sarsa': '#2196F3',     # SARSA (xanh dương)
    'grid': '#BDBDBD'       # Lưới (xám nhạt)
}

# Các ký hiệu hướng
DIRECTION_SYMBOLS = ['↑', '↓', '←', '→']

def visualize_maze(maze: np.ndarray, ax=None, show_grid: bool = True, title: str = "Mê cung") -> None:
    """
    Hiển thị mê cung.

    Args:
        maze: Ma trận biểu diễn mê cung
        ax: Matplotlib axes (tùy chọn)
        show_grid: Hiển thị lưới hay không
        title: Tiêu đề của biểu đồ
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Tạo colormap
    cmap = colors.ListedColormap([COLORS['path'], COLORS['wall'], COLORS['start'], COLORS['goal']])
    
    # Hiển thị mê cung
    ax.imshow(maze, cmap=cmap)
    
    # Thêm lưới nếu cần
    if show_grid:
        ax.grid(color=COLORS['grid'], linestyle='-', linewidth=0.5)
    
    # Thêm tiêu đề
    ax.set_title(title, fontsize=14)
    
    # Thêm nhãn trục
    ax.set_xticks(np.arange(maze.shape[1]))
    ax.set_yticks(np.arange(maze.shape[0]))
    
    # Ẩn giá trị trên trục
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    return ax

def visualize_policy(maze: np.ndarray, policy: np.ndarray, ax=None, title: str = "Chính sách tốt nhất") -> None:
    """
    Hiển thị chính sách trên mê cung.

    Args:
        maze: Ma trận biểu diễn mê cung
        policy: Ma trận chính sách (hành động tốt nhất cho mỗi ô)
        ax: Matplotlib axes (tùy chọn)
        title: Tiêu đề của biểu đồ
    """
    # Hiển thị mê cung
    ax = visualize_maze(maze, ax, title=title)
    
    # Thêm mũi tên biểu thị chính sách
    height, width = maze.shape
    for r in range(height):
        for c in range(width):
            if maze[r, c] == 0:  # Chỉ hiển thị chính sách ở các ô đường đi
                ax.text(c, r, DIRECTION_SYMBOLS[policy[r, c]], 
                        ha='center', va='center', color='blue', fontsize=12, fontweight='bold')
    
    return ax

def visualize_value_function(maze: np.ndarray, value_function: np.ndarray, ax=None, title: str = "Hàm giá trị") -> None:
    """
    Hiển thị hàm giá trị trên mê cung.

    Args:
        maze: Ma trận biểu diễn mê cung
        value_function: Ma trận giá trị
        ax: Matplotlib axes (tùy chọn)
        title: Tiêu đề của biểu đồ
    """
    # Hiển thị mê cung
    ax = visualize_maze(maze, ax, title=title)
    
    # Thêm giá trị
    height, width = maze.shape
    for r in range(height):
        for c in range(width):
            if maze[r, c] == 0:  # Chỉ hiển thị giá trị ở các ô đường đi
                value = value_function[r, c]
                color = 'green' if value > 0 else 'red'
                ax.text(c, r, f"{value:.1f}", 
                        ha='center', va='center', color=color, fontsize=8)
    
    return ax

def visualize_state_visits(maze: np.ndarray, state_visits: np.ndarray, ax=None, title: str = "Lượt thăm trạng thái") -> None:
    """
    Hiển thị số lần thăm mỗi trạng thái.

    Args:
        maze: Ma trận biểu diễn mê cung
        state_visits: Ma trận số lượt thăm
        ax: Matplotlib axes (tùy chọn)
        title: Tiêu đề của biểu đồ
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Tạo bản sao của maze để hiển thị
    visit_map = np.zeros_like(maze, dtype=float)
    
    # Gán giá trị lượt thăm
    height, width = maze.shape
    for r in range(height):
        for c in range(width):
            if maze[r, c] == 0:  # Chỉ xét các ô đường đi
                visit_map[r, c] = state_visits[r, c]
            else:
                visit_map[r, c] = -1  # Đánh dấu các ô tường
    
    # Tạo colormap
    visit_cmap = plt.cm.Blues
    visit_cmap.set_bad(COLORS['wall'])
    
    # Hiển thị heatmap
    im = ax.imshow(visit_map, cmap=visit_cmap)
    
    # Thêm colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Số lần thăm')
    
    # Đánh dấu điểm bắt đầu và điểm đích
    start_pos = np.where(maze == 2)
    goal_pos = np.where(maze == 3)
    
    ax.add_patch(Rectangle((goal_pos[1][0] - 0.5, goal_pos[0][0] - 0.5), 1, 1, 
                          fill=True, color=COLORS['goal'], alpha=0.7))
    ax.add_patch(Rectangle((start_pos[1][0] - 0.5, start_pos[0][0] - 0.5), 1, 1, 
                          fill=True, color=COLORS['start'], alpha=0.7))
    
    # Thêm text cho các ô
    for r in range(height):
        for c in range(width):
            if maze[r, c] == 0 and state_visits[r, c] > 0:  # Chỉ hiển thị giá trị ở các ô đã thăm
                ax.text(c, r, f"{int(state_visits[r, c])}", 
                        ha='center', va='center', color='black', fontsize=8)
    
    # Thêm tiêu đề
    ax.set_title(title, fontsize=14)
    
    # Thêm lưới
    ax.grid(color=COLORS['grid'], linestyle='-', linewidth=0.5)
    
    return ax

def visualize_q_values(maze: np.ndarray, q_table: np.ndarray, position: Tuple[int, int], ax=None, title: str = "Giá trị Q") -> None:
    """
    Hiển thị giá trị Q cho một vị trí cụ thể.

    Args:
        maze: Ma trận biểu diễn mê cung
        q_table: Q-table (shape: [height, width, action_size])
        position: Vị trí (row, col) cần hiển thị giá trị Q
        ax: Matplotlib axes (tùy chọn)
        title: Tiêu đề của biểu đồ
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    row, col = position
    q_values = q_table[row, col]
    action_names = ['Lên', 'Xuống', 'Trái', 'Phải']
    
    # Tạo biểu đồ cột
    bars = ax.bar(action_names, q_values, color=COLORS['q_learning'])
    
    # Thêm giá trị lên mỗi cột
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{q_values[i]:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # Đánh dấu hành động tốt nhất
    best_action = np.argmax(q_values)
    bars[best_action].set_color(COLORS['goal'])
    
    # Thêm tiêu đề và nhãn
    ax.set_title(f"{title} tại vị trí ({row}, {col})", fontsize=14)
    ax.set_ylabel('Giá trị Q')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return ax

def visualize_training_progress(history: Dict[str, List], ax=None, title: str = "Tiến trình huấn luyện", 
                              window_size: int = 10, log_scale: bool = False) -> None:
    """
    Hiển thị tiến trình huấn luyện.

    Args:
        history: Dictionary chứa lịch sử huấn luyện (rewards, steps)
        ax: Matplotlib axes (tùy chọn)
        title: Tiêu đề của biểu đồ
        window_size: Kích thước cửa sổ cho việc làm mịn
        log_scale: Sử dụng thang logarit cho trục y
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    rewards = np.array(history["rewards"])
    steps = np.array(history["steps"])
    episodes = np.arange(1, len(rewards) + 1)
    
    # Tính trung bình di động
    if len(rewards) > window_size:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
        smoothed_episodes = episodes[window_size-1:]
    else:
        smoothed_rewards = rewards
        smoothed_steps = steps
        smoothed_episodes = episodes
    
    # Vẽ biểu đồ
    ax.plot(episodes, rewards, 'o', alpha=0.3, color=COLORS['q_learning'], label='Phần thưởng')
    ax.plot(smoothed_episodes, smoothed_rewards, '-', linewidth=2, color=COLORS['q_learning'], label=f'Phần thưởng (MA{window_size})')
    
    # Thêm trục y thứ hai cho số bước
    ax2 = ax.twinx()
    ax2.plot(episodes, steps, 'o', alpha=0.3, color=COLORS['sarsa'], label='Số bước')
    ax2.plot(smoothed_episodes, smoothed_steps, '-', linewidth=2, color=COLORS['sarsa'], label=f'Số bước (MA{window_size})')
    
    # Thêm tiêu đề và nhãn
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Phần thưởng')
    ax2.set_ylabel('Số bước')
    
    # Thang logarit nếu cần
    if log_scale:
        ax.set_yscale('symlog')
        ax2.set_yscale('symlog')
    
    # Thêm lưới
    ax.grid(linestyle='--', alpha=0.7)
    
    # Kết hợp legend từ cả hai trục
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    return ax, ax2

def visualize_training_results(training_results: Dict[str, List], agent_type: str, 
                              maze_size: Tuple[int, int], save_dir: Optional[str] = None,
                              show_plot: bool = True) -> None:
    """
    Hiển thị kết quả huấn luyện và lưu thành file.

    Args:
        training_results: Kết quả huấn luyện (rewards, steps)
        agent_type: Loại agent ("q_learning" hoặc "sarsa")
        maze_size: Kích thước mê cung (height, width)
        save_dir: Thư mục lưu biểu đồ
        show_plot: Hiển thị biểu đồ
    """
    height, width = maze_size
    
    # Tạo biểu đồ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Hiển thị tiến trình huấn luyện
    visualize_training_progress(training_results, ax=ax1, 
                               title=f"Tiến trình huấn luyện {agent_type.upper()} - Mê cung {height}x{width}")
    
    # Hiển thị biểu đồ hội tụ
    rewards = np.array(training_results["rewards"])
    steps = np.array(training_results["steps"])
    episodes = np.arange(1, len(rewards) + 1)
    
    # Tính trung bình di động cho biểu đồ hội tụ (window_size lớn hơn)
    window_size = min(50, len(rewards) // 10) if len(rewards) > 100 else 10
    
    if len(rewards) > window_size:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        smoothed_episodes = episodes[window_size-1:]
        
        # Vẽ biểu đồ hội tụ
        ax2.plot(smoothed_episodes, smoothed_rewards, '-', linewidth=2, color=COLORS['q_learning'])
        
        # Tính và vẽ độ lệch chuẩn
        std_window = window_size * 2
        reward_std = []
        
        for i in range(window_size-1, len(rewards), std_window):
            end_idx = min(i + std_window, len(rewards))
            std = np.std(rewards[i:end_idx])
            reward_std.append(std)
        
        std_episodes = np.arange(window_size-1, len(rewards), std_window)[:len(reward_std)]
        
        ax2.plot(std_episodes, reward_std, 'o-', color=COLORS['sarsa'], label='Độ lệch chuẩn')
        
        # Thêm đường xu hướng
        z = np.polyfit(smoothed_episodes[-100:] if len(smoothed_episodes) > 100 else smoothed_episodes, 
                      smoothed_rewards[-100:] if len(smoothed_rewards) > 100 else smoothed_rewards, 1)
        p = np.poly1d(z)
        trend_episodes = smoothed_episodes[-100:] if len(smoothed_episodes) > 100 else smoothed_episodes
        ax2.plot(trend_episodes, p(trend_episodes), '--', color='red', label='Xu hướng')
    
    else:
        ax2.plot(episodes, rewards, '-', linewidth=2, color=COLORS['q_learning'])
    
    # Thêm tiêu đề và nhãn
    ax2.set_title(f"Phân tích hội tụ - {agent_type.upper()}", fontsize=14)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Phần thưởng trung bình')
    ax2.grid(linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Thêm thông tin thống kê
    textstr = '\n'.join((
        f'Phần thưởng trung bình: {np.mean(rewards[-100:]):.2f}',
        f'Số bước trung bình: {np.mean(steps[-100:]):.2f}',
        f'Độ lệch chuẩn phần thưởng: {np.std(rewards[-100:]):.2f}',
        f'Độ lệch chuẩn số bước: {np.std(steps[-100:]):.2f}',
    ))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax2.text(0.05, 0.05, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    # Chỉnh layout và lưu biểu đồ
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{agent_type}_{height}x{width}_training_results.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ tại: {save_path}")
    
    # Hiển thị biểu đồ
    if show_plot:
        plt.show()
    else:
        plt.close()

def compare_agents(q_learning_history: Dict[str, List], sarsa_history: Dict[str, List], 
                  maze_size: Tuple[int, int], save_dir: Optional[str] = None,
                  show_plot: bool = True) -> None:
    """
    So sánh hiệu suất của Q-Learning và SARSA.

    Args:
        q_learning_history: Lịch sử huấn luyện Q-Learning
        sarsa_history: Lịch sử huấn luyện SARSA
        maze_size: Kích thước mê cung (height, width)
        save_dir: Thư mục lưu biểu đồ
        show_plot: Hiển thị biểu đồ
    """
    height, width = maze_size
    
    # Tạo biểu đồ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # So sánh phần thưởng
    q_rewards = np.array(q_learning_history["rewards"])
    sarsa_rewards = np.array(sarsa_history["rewards"])
    
    # Đảm bảo cùng độ dài
    min_len = min(len(q_rewards), len(sarsa_rewards))
    q_rewards = q_rewards[:min_len]
    sarsa_rewards = sarsa_rewards[:min_len]
    episodes = np.arange(1, min_len + 1)
    
    # Tính trung bình di động
    window_size = min(50, min_len // 10) if min_len > 100 else 10
    
    if min_len > window_size:
        q_smoothed = np.convolve(q_rewards, np.ones(window_size)/window_size, mode='valid')
        sarsa_smoothed = np.convolve(sarsa_rewards, np.ones(window_size)/window_size, mode='valid')
        smoothed_episodes = episodes[window_size-1:]
        
        # Vẽ biểu đồ phần thưởng
        ax1.plot(episodes, q_rewards, 'o', alpha=0.2, color=COLORS['q_learning'], label='Q-Learning')
        ax1.plot(smoothed_episodes, q_smoothed, '-', linewidth=2, color=COLORS['q_learning'], label=f'Q-Learning (MA{window_size})')
        
        ax1.plot(episodes, sarsa_rewards, 'o', alpha=0.2, color=COLORS['sarsa'], label='SARSA')
        ax1.plot(smoothed_episodes, sarsa_smoothed, '-', linewidth=2, color=COLORS['sarsa'], label=f'SARSA (MA{window_size})')
    else:
        ax1.plot(episodes, q_rewards, '-', linewidth=2, color=COLORS['q_learning'], label='Q-Learning')
        ax1.plot(episodes, sarsa_rewards, '-', linewidth=2, color=COLORS['sarsa'], label='SARSA')
    
    # Thêm tiêu đề và nhãn
    ax1.set_title(f"So sánh phần thưởng - Mê cung {height}x{width}", fontsize=14)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Phần thưởng')
    ax1.grid(linestyle='--', alpha=0.7)
    ax1.legend()
    
    # So sánh số bước
    q_steps = np.array(q_learning_history["steps"])
    sarsa_steps = np.array(sarsa_history["steps"])
    
    # Đảm bảo cùng độ dài
    q_steps = q_steps[:min_len]
    sarsa_steps = sarsa_steps[:min_len]
    
    if min_len > window_size:
        q_steps_smoothed = np.convolve(q_steps, np.ones(window_size)/window_size, mode='valid')
        sarsa_steps_smoothed = np.convolve(sarsa_steps, np.ones(window_size)/window_size, mode='valid')
        
        # Vẽ biểu đồ số bước
        ax2.plot(episodes, q_steps, 'o', alpha=0.2, color=COLORS['q_learning'], label='Q-Learning')
        ax2.plot(smoothed_episodes, q_steps_smoothed, '-', linewidth=2, color=COLORS['q_learning'], label=f'Q-Learning (MA{window_size})')
        
        ax2.plot(episodes, sarsa_steps, 'o', alpha=0.2, color=COLORS['sarsa'], label='SARSA')
        ax2.plot(smoothed_episodes, sarsa_steps_smoothed, '-', linewidth=2, color=COLORS['sarsa'], label=f'SARSA (MA{window_size})')
    else:
        ax2.plot(episodes, q_steps, '-', linewidth=2, color=COLORS['q_learning'], label='Q-Learning')
        ax2.plot(episodes, sarsa_steps, '-', linewidth=2, color=COLORS['sarsa'], label='SARSA')
    
    # Thêm tiêu đề và nhãn
    ax2.set_title(f"So sánh số bước - Mê cung {height}x{width}", fontsize=14)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Số bước')
    ax2.grid(linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Thêm thông tin thống kê
    q_textstr = '\n'.join((
        f'Q-Learning:',
        f'Phần thưởng trung bình: {np.mean(q_rewards[-100:]):.2f}',
        f'Số bước trung bình: {np.mean(q_steps[-100:]):.2f}',
    ))
    
    sarsa_textstr = '\n'.join((
        f'SARSA:',
        f'Phần thưởng trung bình: {np.mean(sarsa_rewards[-100:]):.2f}',
        f'Số bước trung bình: {np.mean(sarsa_steps[-100:]):.2f}',
    ))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax1.text(0.05, 0.05, q_textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    ax2.text(0.05, 0.05, sarsa_textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    # Chỉnh layout và lưu biểu đồ
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"comparison_{height}x{width}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ so sánh tại: {save_path}")
    
    # Hiển thị biểu đồ
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_comprehensive_report(agent, env, training_results: Dict[str, List], 
                               save_dir: Optional[str] = None, agent_type: str = "Q-Learning") -> None:
    """
    Tạo báo cáo tổng hợp về hiệu suất của agent.

    Args:
        agent: Agent học tăng cường
        env: Môi trường mê cung
        training_results: Kết quả huấn luyện
        save_dir: Thư mục lưu báo cáo
        agent_type: Loại agent ("Q-Learning" hoặc "SARSA")
    """
    maze = env.maze
    height, width = maze.shape
    
    # Tạo figure với nhiều subplot
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 3)
    
    # 1. Mê cung với chính sách
    ax1 = fig.add_subplot(gs[0, 0])
    policy = agent.get_policy()
    visualize_policy(maze, policy, ax=ax1, title=f"Chính sách {agent_type}")
    
    # 2. Hàm giá trị
    ax2 = fig.add_subplot(gs[0, 1])
    value_function = agent.get_value_function()
    visualize_value_function(maze, value_function, ax=ax2, title=f"Hàm giá trị {agent_type}")
    
    # 3. Lượt thăm trạng thái
    ax3 = fig.add_subplot(gs[0, 2])
    visualize_state_visits(maze, agent.state_visits, ax=ax3, title=f"Lượt thăm trạng thái {agent_type}")
    
    # 4. Tiến trình huấn luyện
    ax4 = fig.add_subplot(gs[1, :2])
    visualize_training_progress(training_results, ax=ax4, title=f"Tiến trình huấn luyện {agent_type} - Mê cung {height}x{width}")
    
    # 5. Thông tin thống kê
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    rewards = np.array(training_results["rewards"])
    steps = np.array(training_results["steps"])
    
    # Tính các chỉ số quan trọng
    final_rewards = rewards[-100:]
    final_steps = steps[-100:]
    
    stats = {
        'Chỉ số': [
            'Phần thưởng trung bình (100 episode cuối)',
            'Số bước trung bình (100 episode cuối)',
            'Độ lệch chuẩn phần thưởng',
            'Độ lệch chuẩn số bước',
            'Phần thưởng tốt nhất',
            'Số bước ít nhất',
            'Tỷ lệ khám phá cuối cùng (epsilon)',
            'Tổng số episode huấn luyện'
        ],
        'Giá trị': [
            f"{np.mean(final_rewards):.2f}",
            f"{np.mean(final_steps):.2f}",
            f"{np.std(final_rewards):.2f}",
            f"{np.std(final_steps):.2f}",
            f"{np.max(rewards):.2f}",
            f"{np.min(steps[steps > 0])}",
            f"{agent.epsilon:.4f}",
            f"{len(rewards)}"
        ]
    }
    
    # Tạo bảng thống kê
    table = ax5.table(cellText=list(zip(stats['Chỉ số'], stats['Giá trị'])),
                     colWidths=[0.7, 0.3],
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Thêm tiêu đề cho bảng
    ax5.set_title(f"Thống kê huấn luyện {agent_type}", fontsize=14)
    
    # Chỉnh layout và lưu
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{agent_type.lower()}_report_{height}x{width}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu báo cáo tổng hợp tại: {save_path}")
    
    plt.show()