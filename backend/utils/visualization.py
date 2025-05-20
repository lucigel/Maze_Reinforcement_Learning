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
    'dqn': '#9C27B0',       # DQN (tím)
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
    
    cmap = colors.ListedColormap([COLORS['path'], COLORS['wall'], COLORS['start'], COLORS['goal']])

    ax.imshow(maze, cmap=cmap)

    if show_grid:
        ax.grid(color=COLORS['grid'], linestyle='-', linewidth=0.5)


    ax.set_title(title, fontsize=14)
    ax.set_xticks(np.arange(maze.shape[1]))
    ax.set_yticks(np.arange(maze.shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    return ax

def visualize_maze_with_path(maze: np.ndarray, path: List[Tuple[int, int]], ax=None, title: str = "Maze with Path") -> plt.Axes:
    """
    Hiển thị mê cung với đường đi.
    
    Args:
        maze: Ma trận biểu diễn mê cung
        path: Danh sách các trạng thái trên đường đi
        ax: Matplotlib axes (tùy chọn)
        title: Tiêu đề của biểu đồ
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Hiển thị mê cung
    cmap = plt.cm.binary
    ax.imshow(maze, cmap=cmap)
    
    # Vẽ đường đi
    path_x = [p[1] for p in path]
    path_y = [p[0] for p in path]
    ax.plot(path_x, path_y, 'r-', linewidth=2)
    
    # Đánh dấu điểm bắt đầu và kết thúc
    ax.plot(path_x[0], path_y[0], 'go', markersize=10)  # Điểm bắt đầu (màu xanh)
    ax.plot(path_x[-1], path_y[-1], 'ro', markersize=10)  # Điểm kết thúc (màu đỏ)
    
    ax.set_title(title)
    ax.set_xticks(np.arange(maze.shape[1]))
    ax.set_yticks(np.arange(maze.shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    
    return ax

def visualize_heatmap(state_visits: np.ndarray, maze: np.ndarray, ax=None, title: str = "State Visitation Heatmap") -> plt.Axes:
    """
    Hiển thị heatmap số lần thăm các trạng thái.

    Args:
        state_visits: Ma trận số lần thăm mỗi trạng thái
        maze: Ma trận biểu diễn mê cung
        ax: Matplotlib axes (tùy chọn)
        title: Tiêu đề của biểu đồ
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Tạo bản sao của state_visits
    visit_map = state_visits.copy()
    
    # Đánh dấu các ô tường với giá trị âm
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if maze[r, c] == 1:  # Nếu là tường
                visit_map[r, c] = -1
    
    # Tạo colormap với màu tường là xám đậm
    cmap = plt.cm.Blues.copy()
    cmap.set_bad('gray')
    
    # Chuyển các giá trị âm thành giá trị NaN để dùng màu "bad"
    masked_visit_map = np.ma.masked_where(visit_map < 0, visit_map)
    
    # Vẽ heatmap
    im = ax.imshow(masked_visit_map, cmap=cmap)
    
    # Thêm colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Số lần thăm')
    
    # Đánh dấu điểm bắt đầu và kết thúc
    start_pos = np.where(maze == 2)
    goal_pos = np.where(maze == 3)
    
    if len(start_pos[0]) > 0:
        ax.plot(start_pos[1][0], start_pos[0][0], 'go', markersize=10)  # Điểm bắt đầu (màu xanh)
    
    if len(goal_pos[0]) > 0:
        ax.plot(goal_pos[1][0], goal_pos[0][0], 'ro', markersize=10)  # Điểm kết thúc (màu đỏ)
    
    # Thêm số lượt thăm lên các ô
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if maze[r, c] != 1 and state_visits[r, c] > 0:  # Không phải tường và đã thăm
                ax.text(c, r, f"{int(state_visits[r, c])}", 
                        ha='center', va='center', color='black', fontsize=8)
    
    ax.set_title(title)
    ax.grid(True)
    
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
    
    visit_cmap = plt.cm.Blues
    visit_cmap.set_bad(COLORS['wall'])
    
    im = ax.imshow(visit_map, cmap=visit_cmap)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Số lần thăm')
    
    start_pos = np.where(maze == 2)
    goal_pos = np.where(maze == 3)
    
    if len(start_pos[0]) > 0:
        ax.add_patch(Rectangle((start_pos[1][0] - 0.5, start_pos[0][0] - 0.5), 1, 1, 
                            fill=True, color=COLORS['start'], alpha=0.7))
    
    if len(goal_pos[0]) > 0:
        ax.add_patch(Rectangle((goal_pos[1][0] - 0.5, goal_pos[0][0] - 0.5), 1, 1, 
                            fill=True, color=COLORS['goal'], alpha=0.7))
    
    for r in range(height):
        for c in range(width):
            if maze[r, c] == 0 and state_visits[r, c] > 0:  # Chỉ hiển thị giá trị ở các ô đã thăm
                ax.text(c, r, f"{int(state_visits[r, c])}", 
                        ha='center', va='center', color='black', fontsize=8)

    ax.set_title(title, fontsize=14)
    
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
                              window_size: int = 10, log_scale: bool = False, 
                              loss_key: Optional[str] = None) -> None:
    """
    Hiển thị tiến trình huấn luyện.

    Args:
        history: Dictionary chứa lịch sử huấn luyện (rewards, steps, và có thể cả losses)
        ax: Matplotlib axes (tùy chọn)
        title: Tiêu đề của biểu đồ
        window_size: Kích thước cửa sổ cho việc làm mịn
        log_scale: Sử dụng thang logarit cho trục y
        loss_key: Key cho dữ liệu loss trong history (nếu có)
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
    
    # Thêm biểu đồ loss nếu có dữ liệu loss
    if loss_key and loss_key in history and len(history[loss_key]) > 0:
        ax3 = None
        if isinstance(ax, np.ndarray) and len(ax) > 2:
            ax3 = ax[2]
        else:
            # Tạo subplot mới nếu không có sẵn
            fig2, ax3 = plt.subplots(figsize=(12, 6))
        
        visualize_dqn_loss(history[loss_key], ax=ax3, title="Loss DQN")
    
    return ax, ax2

def visualize_dqn_loss(loss_history: List[float], ax=None, title: str = "Loss DQN", 
                     save_dir: Optional[str] = None, show_plot: bool = True) -> None:
    """
    Hiển thị đồ thị loss của DQN.
    
    Args:
        loss_history: Lịch sử giá trị loss
        ax: Matplotlib axes (tùy chọn)
        title: Tiêu đề của biểu đồ
        save_dir: Thư mục lưu biểu đồ
        show_plot: Hiển thị biểu đồ
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Giảm nhiễu bằng cách lấy mẫu nếu quá nhiều điểm
    loss_samples = len(loss_history)
    if loss_samples > 10000:
        sample_rate = loss_samples // 1000
        loss_x = np.arange(0, loss_samples, sample_rate)
        loss_y = np.array([np.mean(loss_history[i:i+sample_rate]) for i in range(0, loss_samples, sample_rate)])
    else:
        loss_x = np.arange(loss_samples)
        loss_y = np.array(loss_history)
    
    # Áp dụng moving average để làm mịn
    window_size = min(100, len(loss_y) // 10) if len(loss_y) > 100 else 10
    if window_size > 1:
        loss_y_smooth = np.convolve(loss_y, np.ones(window_size)/window_size, mode='valid')
        loss_x_smooth = loss_x[window_size-1:]
    else:
        loss_y_smooth = loss_y
        loss_x_smooth = loss_x
    
    # Vẽ biểu đồ
    ax.plot(loss_x_smooth, loss_y_smooth, color=COLORS['dqn'], linewidth=2)
    
    # Thêm tiêu đề và nhãn
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Bước huấn luyện')
    ax.set_ylabel('Loss')
    ax.grid(linestyle='--', alpha=0.7)
    
    # Đặt trục y sang dạng log nếu lớn hơn 0
    if np.max(loss_y_smooth) > 0:
        ax.set_yscale('log')
    
    # Lưu biểu đồ nếu cần
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"dqn_loss.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ loss tại: {save_path}")
    
    # Hiển thị biểu đồ
    if show_plot and ax.figure is not fig:  # Kiểm tra nếu ax là subplot của figure khác
        plt.show()
    elif show_plot:
        plt.tight_layout()
    
    return ax

def visualize_training_results(training_results: Dict[str, List], agent_type: str, 
                              maze_size: Tuple[int, int], save_dir: Optional[str] = None,
                              show_plot: bool = True, loss_key: Optional[str] = None) -> None:
    """
    Hiển thị kết quả huấn luyện và lưu thành file.

    Args:
        training_results: Kết quả huấn luyện (rewards, steps, và có thể cả losses)
        agent_type: Loại agent ("q_learning", "sarsa" hoặc "dqn")
        maze_size: Kích thước mê cung (height, width)
        save_dir: Thư mục lưu biểu đồ
        show_plot: Hiển thị biểu đồ
        loss_key: Key để truy cập dữ liệu loss (nếu có, thường là 'losses' cho DQN)
    """
    height, width = maze_size
    
    # Xác định số lượng subplot dựa vào việc có dữ liệu loss hay không
    has_loss = loss_key is not None and loss_key in training_results and len(training_results[loss_key]) > 0
    
    if has_loss:
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    else:
        fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    # Hiển thị tiến trình huấn luyện (rewards và steps)
    ax1, ax2 = visualize_training_progress(
        training_results, 
        ax=axs[0], 
        title=f"Tiến trình huấn luyện {agent_type.upper()} - Mê cung {height}x{width}"
    )
    
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
        axs[1].plot(smoothed_episodes, smoothed_rewards, '-', linewidth=2, color=COLORS[agent_type])
        
        # Tính và vẽ độ lệch chuẩn
        std_window = window_size * 2
        reward_std = []
        
        for i in range(window_size-1, len(rewards), std_window):
            end_idx = min(i + std_window, len(rewards))
            std = np.std(rewards[i:end_idx])
            reward_std.append(std)
        
        std_episodes = np.arange(window_size-1, len(rewards), std_window)[:len(reward_std)]
        
        if len(std_episodes) > 0 and len(reward_std) > 0:
            axs[1].plot(std_episodes, reward_std, 'o-', color='gray', label='Độ lệch chuẩn')
        
        # Thêm đường xu hướng
        if len(smoothed_episodes) > 0:
            trend_episodes = smoothed_episodes[-100:] if len(smoothed_episodes) > 100 else smoothed_episodes
            trend_rewards = smoothed_rewards[-100:] if len(smoothed_rewards) > 100 else smoothed_rewards
            
            if len(trend_episodes) > 1:  # Cần ít nhất 2 điểm để vẽ đường xu hướng
                z = np.polyfit(trend_episodes, trend_rewards, 1)
                p = np.poly1d(z)
                axs[1].plot(trend_episodes, p(trend_episodes), '--', color='red', label='Xu hướng')
    
    else:
        axs[1].plot(episodes, rewards, '-', linewidth=2, color=COLORS[agent_type])
    
    # Thêm tiêu đề và nhãn
    axs[1].set_title(f"Phân tích hội tụ - {agent_type.upper()}", fontsize=14)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Phần thưởng trung bình')
    axs[1].grid(linestyle='--', alpha=0.7)
    axs[1].legend()
    
    # Thêm thông tin thống kê
    textstr = '\n'.join((
        f'Phần thưởng trung bình: {np.mean(rewards[-100:]):.2f}',
        f'Số bước trung bình: {np.mean(steps[-100:]):.2f}',
        f'Độ lệch chuẩn phần thưởng: {np.std(rewards[-100:]):.2f}',
        f'Độ lệch chuẩn số bước: {np.std(steps[-100:]):.2f}',
    ))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    axs[1].text(0.05, 0.05, textstr, transform=axs[1].transAxes, fontsize=10,
              verticalalignment='bottom', bbox=props)
    
    # Hiển thị biểu đồ loss nếu có dữ liệu loss
    if has_loss:
        visualize_dqn_loss(
            training_results[loss_key], 
            ax=axs[2], 
            title=f"Loss {agent_type.upper()} - Mê cung {height}x{width}",
            show_plot=False
        )
    
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

def compare_all_agents(q_learning_history: Dict[str, List], sarsa_history: Dict[str, List],
                      dqn_history: Dict[str, List], maze_size: Tuple[int, int], 
                      save_dir: Optional[str] = None, show_plot: bool = True,
                      loss_key: Optional[str] = None) -> None:
    """
    So sánh hiệu suất của Q-Learning, SARSA và DQN.
    
    Args:
        q_learning_history: Lịch sử huấn luyện Q-Learning
        sarsa_history: Lịch sử huấn luyện SARSA
        dqn_history: Lịch sử huấn luyện DQN
        maze_size: Kích thước mê cung (height, width)
        save_dir: Thư mục lưu biểu đồ
        show_plot: Hiển thị biểu đồ
        loss_key: Key để truy cập dữ liệu loss trong DQN history (thường là 'losses')
    """
    height, width = maze_size
    
    # Tạo biểu đồ
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    
    # So sánh phần thưởng
    q_rewards = np.array(q_learning_history["rewards"])
    sarsa_rewards = np.array(sarsa_history["rewards"])
    dqn_rewards = np.array(dqn_history["rewards"])
    
    # Đảm bảo cùng độ dài
    min_len = min(len(q_rewards), len(sarsa_rewards), len(dqn_rewards))
    q_rewards = q_rewards[:min_len]
    sarsa_rewards = sarsa_rewards[:min_len]
    dqn_rewards = dqn_rewards[:min_len]
    episodes = np.arange(1, min_len + 1)
    
    # Tính trung bình di động
    window_size = min(50, min_len // 10) if min_len > 100 else 10
    
    if min_len > window_size:
        q_smoothed = np.convolve(q_rewards, np.ones(window_size)/window_size, mode='valid')
        sarsa_smoothed = np.convolve(sarsa_rewards, np.ones(window_size)/window_size, mode='valid')
        dqn_smoothed = np.convolve(dqn_rewards, np.ones(window_size)/window_size, mode='valid')
        smoothed_episodes = episodes[window_size-1:]
        
        # Vẽ biểu đồ phần thưởng
        axs[0, 0].plot(smoothed_episodes, q_smoothed, '-', linewidth=2, color=COLORS['q_learning'], label='Q-Learning')
        axs[0, 0].plot(smoothed_episodes, sarsa_smoothed, '-', linewidth=2, color=COLORS['sarsa'], label='SARSA')
        axs[0, 0].plot(smoothed_episodes, dqn_smoothed, '-', linewidth=2, color=COLORS['dqn'], label='DQN')
    else:
        axs[0, 0].plot(episodes, q_rewards, '-', linewidth=2, color=COLORS['q_learning'], label='Q-Learning')
        axs[0, 0].plot(episodes, sarsa_rewards, '-', linewidth=2, color=COLORS['sarsa'], label='SARSA')
        axs[0, 0].plot(episodes, dqn_rewards, '-', linewidth=2, color=COLORS['dqn'], label='DQN')
    
    # Thêm tiêu đề và nhãn
    axs[0, 0].set_title(f"So sánh phần thưởng - Mê cung {height}x{width}", fontsize=14)
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Phần thưởng')
    axs[0, 0].grid(linestyle='--', alpha=0.7)
    axs[0, 0].legend()
    
    # So sánh số bước
    q_steps = np.array(q_learning_history["steps"])
    sarsa_steps = np.array(sarsa_history["steps"])
    dqn_steps = np.array(dqn_history["steps"])
    
    # Đảm bảo cùng độ dài
    q_steps = q_steps[:min_len]
    sarsa_steps = sarsa_steps[:min_len]
    dqn_steps = dqn_steps[:min_len]
    
    if min_len > window_size:
        q_steps_smoothed = np.convolve(q_steps, np.ones(window_size)/window_size, mode='valid')
        sarsa_steps_smoothed = np.convolve(sarsa_steps, np.ones(window_size)/window_size, mode='valid')
        dqn_steps_smoothed = np.convolve(dqn_steps, np.ones(window_size)/window_size, mode='valid')
        
        # Vẽ biểu đồ số bước
        axs[0, 1].plot(smoothed_episodes, q_steps_smoothed, '-', linewidth=2, color=COLORS['q_learning'], label='Q-Learning')
        axs[0, 1].plot(smoothed_episodes, sarsa_steps_smoothed, '-', linewidth=2, color=COLORS['sarsa'], label='SARSA')
        axs[0, 1].plot(smoothed_episodes, dqn_steps_smoothed, '-', linewidth=2, color=COLORS['dqn'], label='DQN')
    else:
        axs[0, 1].plot(episodes, q_steps, '-', linewidth=2, color=COLORS['q_learning'], label='Q-Learning')
        axs[0, 1].plot(episodes, sarsa_steps, '-', linewidth=2, color=COLORS['sarsa'], label='SARSA')
        axs[0, 1].plot(episodes, dqn_steps, '-', linewidth=2, color=COLORS['dqn'], label='DQN')
    
    # Thêm tiêu đề và nhãn
    axs[0, 1].set_title(f"So sánh số bước - Mê cung {height}x{width}", fontsize=14)
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Số bước')
    axs[0, 1].grid(linestyle='--', alpha=0.7)
    axs[0, 1].legend()
    
    # So sánh tỷ lệ thành công (từ episode 100 trở đi, 100 episode cuối)
    start_idx = 100 if min_len > 200 else min_len // 2
    end_idx = min_len
    
    # Tính tỷ lệ thành công (được định nghĩa là steps < max_steps hoặc 100)
    max_steps = 100  # Giả định max_steps là 100, điều chỉnh nếu cần
    
    q_success = np.sum(q_steps[start_idx:end_idx] < max_steps) / (end_idx - start_idx)
    sarsa_success = np.sum(sarsa_steps[start_idx:end_idx] < max_steps) / (end_idx - start_idx)
    dqn_success = np.sum(dqn_steps[start_idx:end_idx] < max_steps) / (end_idx - start_idx)
    
    # Tính số bước trung bình (chỉ tính các episode thành công)
    q_successful_steps = q_steps[start_idx:end_idx][q_steps[start_idx:end_idx] < max_steps]
    sarsa_successful_steps = sarsa_steps[start_idx:end_idx][sarsa_steps[start_idx:end_idx] < max_steps]
    dqn_successful_steps = dqn_steps[start_idx:end_idx][dqn_steps[start_idx:end_idx] < max_steps]
    
    q_avg_steps = np.mean(q_successful_steps) if len(q_successful_steps) > 0 else 0
    sarsa_avg_steps = np.mean(sarsa_successful_steps) if len(sarsa_successful_steps) > 0 else 0
    dqn_avg_steps = np.mean(dqn_successful_steps) if len(dqn_successful_steps) > 0 else 0
    
    # Vẽ biểu đồ cột
    algorithms = ['Q-Learning', 'SARSA', 'DQN']
    success_rates = [q_success * 100, sarsa_success * 100, dqn_success * 100]
    avg_steps = [q_avg_steps, sarsa_avg_steps, dqn_avg_steps]
    
    # Biểu đồ tỷ lệ thành công
    axs[1, 0].bar(algorithms, success_rates, color=[COLORS['q_learning'], COLORS['sarsa'], COLORS['dqn']])
    axs[1, 0].set_title('Tỷ lệ thành công', fontsize=14)
    axs[1, 0].set_ylabel('Tỷ lệ thành công (%)')
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Thêm giá trị lên mỗi cột
    for i, v in enumerate(success_rates):
        axs[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # Biểu đồ số bước trung bình
    axs[1, 1].bar(algorithms, avg_steps, color=[COLORS['q_learning'], COLORS['sarsa'], COLORS['dqn']])
    axs[1, 1].set_title('Số bước trung bình (episode thành công)', fontsize=14)
    axs[1, 1].set_ylabel('Số bước')
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Thêm giá trị lên mỗi cột
    for i, v in enumerate(avg_steps):
        axs[1, 1].text(i, v + 1, f'{v:.1f}', ha='center')
    
    # Thêm thông tin thống kê
    stats_text = '\n'.join((
        f'Tỷ lệ thành công:',
        f'Q-Learning: {q_success*100:.1f}%',
        f'SARSA: {sarsa_success*100:.1f}%',
        f'DQN: {dqn_success*100:.1f}%',
        f'\nSố bước trung bình:',
        f'Q-Learning: {q_avg_steps:.1f}',
        f'SARSA: {sarsa_avg_steps:.1f}',
        f'DQN: {dqn_avg_steps:.1f}',
    ))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    axs[1, 0].text(0.05, 0.05, stats_text, transform=axs[1, 0].transAxes, fontsize=10,
                 verticalalignment='bottom', bbox=props)
    
    # Chỉnh layout và lưu biểu đồ
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"comparison_all_{height}x{width}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ so sánh tại: {save_path}")
    
    # Hiển thị biểu đồ loss nếu có
    if loss_key and loss_key in dqn_history and len(dqn_history[loss_key]) > 0:
        visualize_dqn_loss(
            dqn_history[loss_key], 
            title=f"Loss DQN - Mê cung {height}x{width}",
            save_dir=save_dir
        )
    
    # Hiển thị biểu đồ
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_comprehensive_report(agent, env, training_results: Dict[str, List], 
                               save_dir: Optional[str] = None, agent_type: str = "q_learning",
                               loss_key: Optional[str] = None) -> None:
    """
    Tạo báo cáo tổng hợp về hiệu suất của agent.

    Args:
        agent: Agent học tăng cường
        env: Môi trường mê cung
        training_results: Kết quả huấn luyện
        save_dir: Thư mục lưu báo cáo
        agent_type: Loại agent ("q_learning", "sarsa" hoặc "dqn")
        loss_key: Key để truy cập dữ liệu loss (nếu có, thường là 'losses' cho DQN)
    """
    maze = env.maze
    height, width = maze.shape
    
    # Xác định số lượng subplot dựa vào việc có dữ liệu loss hay không
    has_loss = loss_key is not None and loss_key in training_results and len(training_results[loss_key]) > 0
    
    if has_loss:
        fig = plt.figure(figsize=(20, 20))
        gs = fig.add_gridspec(3, 3)
    else:
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(2, 3)
    
    # 1. Mê cung với chính sách
    ax1 = fig.add_subplot(gs[0, 0])
    policy = agent.get_policy()
    visualize_policy(maze, policy, ax=ax1, title=f"Chính sách {agent_type.upper()}")
    
    # 2. Hàm giá trị
    ax2 = fig.add_subplot(gs[0, 1])
    value_function = agent.get_value_function()
    visualize_value_function(maze, value_function, ax=ax2, title=f"Hàm giá trị {agent_type.upper()}")
    
    # 3. Lượt thăm trạng thái
    ax3 = fig.add_subplot(gs[0, 2])
    visualize_state_visits(maze, agent.state_visits, ax=ax3, title=f"Lượt thăm trạng thái {agent_type.upper()}")
    
    # 4. Tiến trình huấn luyện
    ax4 = fig.add_subplot(gs[1, :2])
    visualize_training_progress(
        training_results, 
        ax=ax4, 
        title=f"Tiến trình huấn luyện {agent_type.upper()} - Mê cung {height}x{width}"
    )
    
    # 5. Thông tin thống kê
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    rewards = np.array(training_results["rewards"])
    steps = np.array(training_results["steps"])
    
    # Tính các chỉ số quan trọng
    final_rewards = rewards[-100:] if len(rewards) > 100 else rewards
    final_steps = steps[-100:] if len(steps) > 100 else steps
    
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
            f"{np.min(steps[steps > 0])}" if np.any(steps > 0) else "N/A",
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
    ax5.set_title(f"Thống kê huấn luyện {agent_type.upper()}", fontsize=14)
    
    # Nếu có dữ liệu loss
    if has_loss:
        ax6 = fig.add_subplot(gs[2, :])
        visualize_dqn_loss(
            training_results[loss_key], 
            ax=ax6, 
            title=f"Loss {agent_type.upper()} - Mê cung {height}x{width}",
            show_plot=False
        )
    
    # Chỉnh layout và lưu
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{agent_type.lower()}_report_{height}x{width}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu báo cáo tổng hợp tại: {save_path}")
    
    plt.show()

def visualize_path_comparison(env, agent, save_dir=None, show_plot=True, agent_type="q_learning"):
    """
    Hiển thị so sánh giữa đường đi của agent và đường đi ngắn nhất.
    
    Args:
        env: Môi trường mê cung
        agent: Agent đã huấn luyện
        save_dir: Thư mục lưu biểu đồ
        show_plot: Hiển thị biểu đồ hay không
        agent_type: Loại agent ("q_learning", "sarsa" hoặc "dqn")
    """
    # Tìm đường đi của agent
    agent_path = []
    state = env.reset()
    done = False
    steps = 0
    
    # Đặt epsilon = 0 để không khám phá
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    while not done and steps < env.max_steps:
        # Chọn hành động theo chính sách đã học
        if agent_type == "dqn":
            action = agent.choose_action(state, env.maze)
        else:
            action = np.argmax(agent.q_table[state])
        
        # Thực hiện hành động
        next_state, reward, done, _ = env.step(action)
        agent_path.append(state)
        state = next_state
        steps += 1
    
    # Thêm trạng thái cuối cùng
    if not done:
        agent_path.append(state)
    
    # Khôi phục epsilon
    agent.epsilon = original_epsilon
    
    # Tìm đường đi ngắn nhất
    shortest_path = env.get_shortest_path()
    
    # Vẽ cả hai đường đi trên mê cung
    plt.figure(figsize=(10, 10))
    maze_plot = env.maze.copy()
    
    # Hiển thị mê cung
    ax = visualize_maze(maze_plot, title="So sánh đường đi")
    
    # Vẽ đường đi ngắn nhất
    if shortest_path:
        shortest_x = [p[1] for p in shortest_path]
        shortest_y = [p[0] for p in shortest_path]
        plt.plot(shortest_x, shortest_y, 'b-', linewidth=2, label='Đường đi ngắn nhất')
    
    # Vẽ đường đi của agent
    agent_x = [p[1] for p in agent_path]
    agent_y = [p[0] for p in agent_path]
    plt.plot(agent_x, agent_y, 'r--', linewidth=2, label=f'Đường đi của {agent_type.upper()}')
    
    # Thêm thông tin phân tích
    if shortest_path:
        shortest_length = len(shortest_path) - 1
        agent_length = len(agent_path)
        efficiency = shortest_length / agent_length if agent_length > 0 else 0
        
        stats_text = '\n'.join((
            f'Đường đi ngắn nhất: {shortest_length} bước',
            f'Đường đi của agent: {agent_length} bước',
            f'Hiệu suất: {efficiency:.2%}'
        ))
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.gca().text(0.05, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='bottom', bbox=props)
    
    plt.legend()
    plt.grid(True)
    
    # Lưu hình
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{agent_type}_path_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu so sánh đường đi tại: {save_path}")
    
    # Hiển thị biểu đồ
    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_experiment_results(experiment_results: Dict[str, Dict[str, Dict[str, List]]], 
                               maze_sizes: List[Tuple[int, int]], save_dir: Optional[str] = None,
                               show_plot: bool = True) -> None:
    """
    Hiển thị kết quả thí nghiệm cho nhiều kích thước mê cung và nhiều thuật toán.
    
    Args:
        experiment_results: Dictionary với cấu trúc {maze_size: {agent_type: results}}
        maze_sizes: Danh sách các kích thước mê cung đã thử nghiệm
        save_dir: Thư mục lưu biểu đồ
        show_plot: Hiển thị biểu đồ hay không
    """
    # Tạo biểu đồ
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Màu sắc cho các thuật toán
    colors = {
        'q_learning': COLORS['q_learning'],
        'sarsa': COLORS['sarsa'],
        'dqn': COLORS['dqn']
    }
    
    # Chuẩn bị dữ liệu cho từng kích thước mê cung
    maze_labels = [f"{height}x{width}" for height, width in maze_sizes]
    
    # Dữ liệu cho biểu đồ
    reward_data = {agent_type: [] for agent_type in colors.keys()}
    steps_data = {agent_type: [] for agent_type in colors.keys()}
    convergence_data = {agent_type: [] for agent_type in colors.keys()}
    
    # Thu thập dữ liệu từ kết quả thí nghiệm
    for maze_size in maze_sizes:
        if maze_size in experiment_results:
            for agent_type in colors.keys():
                if agent_type in experiment_results[maze_size]:
                    results = experiment_results[maze_size][agent_type]
                    
                    # Lấy 100 episode cuối cùng để đánh giá
                    rewards = np.array(results["rewards"])
                    steps = np.array(results["steps"])
                    
                    last_100_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
                    last_100_steps = steps[-100:] if len(steps) >= 100 else steps
                    
                    # Tính giá trị trung bình
                    avg_reward = np.mean(last_100_rewards)
                    avg_steps = np.mean(last_100_steps)
                    
                    # Tính thời gian hội tụ (số episode cần để đạt được 90% phần thưởng cuối cùng)
                    target_reward = 0.9 * avg_reward
                    try:
                        convergence_episode = np.where(rewards >= target_reward)[0][0]
                    except IndexError:
                        convergence_episode = len(rewards)  # Không hội tụ
                    
                    # Lưu vào dữ liệu biểu đồ
                    reward_data[agent_type].append(avg_reward)
                    steps_data[agent_type].append(avg_steps)
                    convergence_data[agent_type].append(convergence_episode)
                else:
                    # Không có dữ liệu cho agent này
                    reward_data[agent_type].append(0)
                    steps_data[agent_type].append(0)
                    convergence_data[agent_type].append(0)
    
    # Vẽ biểu đồ so sánh phần thưởng
    width = 0.25  # Độ rộng của cột
    x = np.arange(len(maze_labels))
    
    for i, (agent_type, values) in enumerate(reward_data.items()):
        ax1.bar(x + (i - 1) * width, values, width, label=agent_type.upper(), color=colors[agent_type])
    
    ax1.set_title('Phần thưởng trung bình')
    ax1.set_xlabel('Kích thước mê cung')
    ax1.set_ylabel('Phần thưởng')
    ax1.set_xticks(x)
    ax1.set_xticklabels(maze_labels)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Vẽ biểu đồ so sánh số bước
    for i, (agent_type, values) in enumerate(steps_data.items()):
        ax2.bar(x + (i - 1) * width, values, width, label=agent_type.upper(), color=colors[agent_type])
    
    ax2.set_title('Số bước trung bình')
    ax2.set_xlabel('Kích thước mê cung')
    ax2.set_ylabel('Số bước')
    ax2.set_xticks(x)
    ax2.set_xticklabels(maze_labels)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Vẽ biểu đồ so sánh thời gian hội tụ
    for i, (agent_type, values) in enumerate(convergence_data.items()):
        ax3.bar(x + (i - 1) * width, values, width, label=agent_type.upper(), color=colors[agent_type])
    
    ax3.set_title('Thời gian hội tụ (số episode)')
    ax3.set_xlabel('Kích thước mê cung')
    ax3.set_ylabel('Số episode')
    ax3.set_xticks(x)
    ax3.set_xticklabels(maze_labels)
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "experiment_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ so sánh thí nghiệm tại: {save_path}")
    
    # Hiển thị biểu đồ
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_beautiful_summary(experiment_results: Dict[str, Dict[str, Dict[str, List]]], 
                            maze_sizes: List[Tuple[int, int]], algorithms: List[str],
                            save_dir: Optional[str] = None, show_plot: bool = True) -> None:
    """
    Tạo báo cáo tổng hợp đẹp mắt cho đồ án.
    
    Args:
        experiment_results: Dictionary với cấu trúc {maze_size: {agent_type: results}}
        maze_sizes: Danh sách các kích thước mê cung đã thử nghiệm
        algorithms: Danh sách các thuật toán đã thử nghiệm
        save_dir: Thư mục lưu biểu đồ
        show_plot: Hiển thị biểu đồ hay không
    """
    # Tạo biểu đồ tổng hợp
    fig = plt.figure(figsize=(20, 30))
    gs = fig.add_gridspec(5, 2)
    
    # 1. Tiêu đề
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, "REINFORCEMENT LEARNING TRONG GIẢI BÀI TOÁN MÊ CUNG",
                horizontalalignment='center', verticalalignment='center',
                fontsize=24, fontweight='bold')
    
    # 2. Biểu đồ so sánh hiệu suất
    # Chuẩn bị dữ liệu
    maze_labels = [f"{height}x{width}" for height, width in maze_sizes]
    
    # Dữ liệu cho biểu đồ
    reward_data = {agent_type: [] for agent_type in algorithms}
    steps_data = {agent_type: [] for agent_type in algorithms}
    convergence_data = {agent_type: [] for agent_type in algorithms}
    
    # Thu thập dữ liệu từ kết quả thí nghiệm
    for maze_size in maze_sizes:
        if maze_size in experiment_results:
            for agent_type in algorithms:
                if agent_type in experiment_results[maze_size]:
                    results = experiment_results[maze_size][agent_type]
                    
                    # Lấy 100 episode cuối cùng để đánh giá
                    rewards = np.array(results["rewards"])
                    steps = np.array(results["steps"])
                    
                    last_100_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
                    last_100_steps = steps[-100:] if len(steps) >= 100 else steps
                    
                    # Tính giá trị trung bình
                    avg_reward = np.mean(last_100_rewards)
                    avg_steps = np.mean(last_100_steps)
                    
                    # Tính thời gian hội tụ (số episode cần để đạt được 90% phần thưởng cuối cùng)
                    target_reward = 0.9 * avg_reward
                    try:
                        convergence_episode = np.where(rewards >= target_reward)[0][0]
                    except IndexError:
                        convergence_episode = len(rewards)  # Không hội tụ
                    
                    # Lưu vào dữ liệu biểu đồ
                    reward_data[agent_type].append(avg_reward)
                    steps_data[agent_type].append(avg_steps)
                    convergence_data[agent_type].append(convergence_episode)
                else:
                    # Không có dữ liệu cho agent này
                    reward_data[agent_type].append(0)
                    steps_data[agent_type].append(0)
                    convergence_data[agent_type].append(0)
    
    # Biểu đồ so sánh phần thưởng
    ax_reward = fig.add_subplot(gs[1, 0])
    x = np.arange(len(maze_labels))
    width = 0.25
    
    for i, (agent_type, values) in enumerate(reward_data.items()):
        ax_reward.bar(x + (i - 1) * width, values, width, label=agent_type.upper(), color=COLORS[agent_type])
    
    ax_reward.set_title('Phần thưởng trung bình', fontsize=16)
    ax_reward.set_xlabel('Kích thước mê cung', fontsize=12)
    ax_reward.set_ylabel('Phần thưởng', fontsize=12)
    ax_reward.set_xticks(x)
    ax_reward.set_xticklabels(maze_labels)
    ax_reward.legend()
    ax_reward.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Biểu đồ so sánh số bước
    ax_steps = fig.add_subplot(gs[1, 1])
    
    for i, (agent_type, values) in enumerate(steps_data.items()):
        ax_steps.bar(x + (i - 1) * width, values, width, label=agent_type.upper(), color=COLORS[agent_type])
    
    ax_steps.set_title('Số bước trung bình', fontsize=16)
    ax_steps.set_xlabel('Kích thước mê cung', fontsize=12)
    ax_steps.set_ylabel('Số bước', fontsize=12)
    ax_steps.set_xticks(x)
    ax_steps.set_xticklabels(maze_labels)
    ax_steps.legend()
    ax_steps.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Biểu đồ đường cong học tập (chọn mê cung lớn nhất)
    largest_maze = maze_sizes[-1]
    
    # Vẽ biểu đồ đường cong học tập cho từng thuật toán
    for i, agent_type in enumerate(algorithms):
        ax_learning_curve = fig.add_subplot(gs[2, i])
        
        if largest_maze in experiment_results and agent_type in experiment_results[largest_maze]:
            results = experiment_results[largest_maze][agent_type]
            
            rewards = np.array(results["rewards"])
            steps = np.array(results["steps"])
            episodes = np.arange(1, len(rewards) + 1)
            
            # Tính trung bình di động
            window_size = min(50, len(rewards) // 10) if len(rewards) > 100 else 10
            
            if len(rewards) > window_size:
                smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                smoothed_steps = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
                smoothed_episodes = episodes[window_size-1:]
                
                # Vẽ phần thưởng
                ax_learning_curve.plot(smoothed_episodes, smoothed_rewards, '-', 
                                     linewidth=2, color=COLORS[agent_type], label='Phần thưởng')
                
                # Vẽ số bước trên trục y thứ hai
                ax2 = ax_learning_curve.twinx()
                ax2.plot(smoothed_episodes, smoothed_steps, '--', 
                       linewidth=2, color='gray', label='Số bước')
                ax2.set_ylabel('Số bước', fontsize=12)
            
            ax_learning_curve.set_title(f'Đường cong học tập - {agent_type.upper()}', fontsize=16)
            ax_learning_curve.set_xlabel('Episode', fontsize=12)
            ax_learning_curve.set_ylabel('Phần thưởng', fontsize=12)
            ax_learning_curve.grid(linestyle='--', alpha=0.7)
            
            # Kết hợp legend
            lines1, labels1 = ax_learning_curve.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax_learning_curve.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    # 4. Bảng so sánh thuật toán
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis('off')
    
    # Chuẩn bị dữ liệu cho bảng
    table_data = []
    
    # Tiêu đề bảng
    header = ['Thuật toán', 'Phần thưởng trung bình', 'Số bước trung bình', 
             'Thời gian hội tụ', 'Ưu điểm', 'Nhược điểm']
    table_data.append(header)
    
    # Dữ liệu thuật toán
    algo_descriptions = {
        'q_learning': {
            'advantages': 'Off-policy, hội tụ nhanh, ưu tiên hành động tốt nhất',
            'disadvantages': 'Có thể thiên về hành động tối ưu cục bộ'
        },
        'sarsa': {
            'advantages': 'On-policy, an toàn hơn, xem xét chính sách hiện tại',
            'disadvantages': 'Hội tụ chậm hơn Q-Learning, bị phụ thuộc vào epsilon'
        },
        'dqn': {
            'advantages': 'Xử lý không gian trạng thái lớn, khả năng tổng quát cao',
            'disadvantages': 'Yêu cầu tính toán cao, phức tạp hơn, nhiều hyperparameter'
        }
    }
    
    for agent_type in algorithms:
        # Tính giá trị trung bình qua các kích thước mê cung
        avg_reward = np.mean([r for r in reward_data[agent_type] if r > 0])
        avg_steps = np.mean([s for s in steps_data[agent_type] if s > 0])
        avg_convergence = np.mean([c for c in convergence_data[agent_type] if c > 0])
        
        row = [
            agent_type.upper(),
            f"{avg_reward:.2f}",
            f"{avg_steps:.2f}",
            f"{avg_convergence:.0f} episodes",
            algo_descriptions[agent_type]['advantages'],
            algo_descriptions[agent_type]['disadvantages']
        ]
        
        table_data.append(row)
    
    # Tạo bảng
    table = ax_table.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
        colColours=[COLORS.get(algo, 'white') if i == 0 else '#F5F5F5' for i, algo in enumerate(['q_learning', 'sarsa', 'dqn'])] + ['#F5F5F5'] * 3
    )
    
    # Điều chỉnh bảng
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # 5. Kết luận và so sánh
    ax_conclusion = fig.add_subplot(gs[4, :])
    ax_conclusion.axis('off')
    
    # Tìm thuật toán tốt nhất cho từng kích thước mê cung
    best_algo_by_maze = {}
    
    for i, maze_size in enumerate(maze_sizes):
        maze_label = maze_labels[i]
        
        # Tìm thuật toán có phần thưởng cao nhất
        best_reward_algo = max(algorithms, key=lambda a: reward_data[a][i] if i < len(reward_data[a]) else 0)
        
        # Tìm thuật toán có số bước ít nhất
        valid_steps = {a: steps_data[a][i] for a in algorithms if i < len(steps_data[a]) and steps_data[a][i] > 0}
        best_steps_algo = min(valid_steps.items(), key=lambda x: x[1])[0] if valid_steps else None
        
        best_algo_by_maze[maze_label] = {'reward': best_reward_algo, 'steps': best_steps_algo}
    
    # Tạo text kết luận
    conclusion_text = "KẾT LUẬN\n\n"
    
    # Kết luận cho từng kích thước mê cung
    conclusion_text += "Thuật toán phù hợp nhất theo kích thước mê cung:\n"
    
    for maze_label, best_algos in best_algo_by_maze.items():
        conclusion_text += f"• Mê cung {maze_label}:\n"
        conclusion_text += f"  - Phần thưởng tốt nhất: {best_algos['reward'].upper()}\n"
        if best_algos['steps']:
            conclusion_text += f"  - Số bước ít nhất: {best_algos['steps'].upper()}\n"
    
    # Kết luận tổng quát
    conclusion_text += "\nNhận xét tổng quát:\n"
    conclusion_text += "• Q-Learning: Phù hợp với mê cung nhỏ và vừa, hội tụ nhanh.\n"
    conclusion_text += "• SARSA: An toàn hơn trong môi trường có nhiều cạm bẫy, ổn định.\n"
    conclusion_text += "• DQN: Hiệu quả nhất cho mê cung lớn, có khả năng tổng quát hóa tốt.\n\n"
    
    conclusion_text += "Đề xuất sử dụng:\n"
    conclusion_text += "• Mê cung nhỏ (<15x15): Q-Learning\n"
    conclusion_text += "• Mê cung vừa (15x15 - 30x30): SARSA hoặc Q-Learning\n"
    conclusion_text += "• Mê cung lớn (>30x30): DQN\n"
    
    # Hiển thị kết luận
    ax_conclusion.text(0.5, 1.0, conclusion_text, ha='center', va='top', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "reinforcement_learning_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu báo cáo tổng hợp tại: {save_path}")
    
    # Hiển thị biểu đồ
    if show_plot:
        plt.show()
    else:
        plt.close()