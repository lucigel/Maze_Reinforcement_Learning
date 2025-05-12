# training.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, List

# Import các module cần thiết
from backend.maze_generators.dfs_generator import DFSMazeGenerator
from backend.environment.maze_env import MazeEnvironment
from backend.rl_agents.q_learning import QLearningAgent
from backend.utils.config import get_training_config
from backend.utils.visualization import plot_training_progress, plot_q_value_heatmap

def train_agent(maze_size: Tuple[int, int] = (10, 10),
                agent_type: str = "q_learning",
                num_episodes: int = 1000,
                max_steps_per_episode: int = 500,
                save_path: Optional[str] = None,
                save_interval: int = 100,
                render_interval: int = 0,
                verbose: bool = True,
                seed: Optional[int] = None) -> Tuple[Dict[str, List[float]], Any]:
    """
    Huấn luyện một agent học tăng cường trên môi trường mê cung.
    
    Args:
        maze_size (Tuple[int, int]): Kích thước mê cung (height, width)
        agent_type (str): Loại agent ("q_learning" hoặc "sarsa")
        num_episodes (int): Số lượng episode huấn luyện
        max_steps_per_episode (int): Số bước tối đa trong mỗi episode
        save_path (str, optional): Đường dẫn để lưu mô hình
        save_interval (int): Khoảng cách episode để lưu mô hình
        render_interval (int): Khoảng cách episode để hiển thị môi trường (0 để tắt)
        verbose (bool): Hiển thị thông tin trong quá trình huấn luyện
        seed (int, optional): Hạt giống cho tính nhất quán
    
    Returns:
        Tuple[Dict[str, List[float]], Any]: Tuple gồm (lịch sử huấn luyện, agent đã học)
    """
    # Lấy cấu hình từ config
    height, width = maze_size
    config = get_training_config(agent_type, maze_size)
    
    # Tạo bộ sinh mê cung và môi trường
    if verbose:
        print(f"Tạo mê cung kích thước {height}x{width} sử dụng DFS...")
    
    maze_generator = DFSMazeGenerator(height=height, width=width, seed=seed)
    env = MazeEnvironment(maze_generator=maze_generator, 
                         max_steps=max_steps_per_episode,
                         move_reward=config["move_reward"],
                         wall_penalty=config["wall_penalty"],
                         goal_reward=config["goal_reward"],
                         time_penalty=config["time_penalty"])
    
    # Khởi tạo agent
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    
    if agent_type == "q_learning":
        agent = QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=config["learning_rate"],
            discount_factor=config["discount_factor"],
            exploration_rate=config["exploration_rate"],
            exploration_decay=config["exploration_decay"],
            min_exploration_rate=config["min_exploration_rate"],
            seed=seed
        )
    else:  # sarsa
        from backend.rl_agents.sarsa import SARSAAgent
        agent = SARSAAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=config["learning_rate"],
            discount_factor=config["discount_factor"],
            exploration_rate=config["exploration_rate"],
            exploration_decay=config["exploration_decay"],
            min_exploration_rate=config["min_exploration_rate"],
            seed=seed
        )
    
    # Thông tin huấn luyện
    if verbose:
        print(f"Bắt đầu huấn luyện agent {agent_type.upper()} trên mê cung {height}x{width}...")
        print(f"Số episode: {num_episodes}")
        print(f"Epsilon ban đầu: {config['exploration_rate']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Discount factor: {config['discount_factor']}")
    
    # Các biến theo dõi
    training_history = {
        "episode_rewards": [],
        "episode_steps": [],
        "success_rate": [],
        "epsilon_values": []
    }
    
    success_window = []  # Cửa sổ trượt để tính tỷ lệ thành công
    start_time = time.time()
    
    # Vòng lặp huấn luyện
    for episode in range(1, num_episodes + 1):
        # Reset môi trường
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Hiển thị môi trường nếu cần
        if render_interval > 0 and episode % render_interval == 0:
            print(f"\nEpisode {episode}:")
            env.render(mode='console')
        
        # Vòng lặp trong mỗi episode
        step = 0
        while not done:
            # Chọn hành động
            action = agent.choose_action(state)
            
            # Thực hiện hành động
            next_state, reward, done, info = env.step(action)
            
            # Học từ trải nghiệm
            agent.learn(state, action, reward, next_state, done)
            
            # Cập nhật trạng thái và phần thưởng
            state = next_state
            episode_reward += reward
            step += 1
            
            # Hiển thị từng bước nếu cần
            if render_interval > 0 and episode % render_interval == 0:
                env.render(mode='console')
                print(f"Bước {step}: Hành động={info['action']}, "
                      f"Phần thưởng={reward:.1f}, Tổng phần thưởng={episode_reward:.1f}")
                time.sleep(0.1)  # Làm chậm để quan sát
        
        # Giảm tỷ lệ khám phá
        agent.decay_exploration()
        
        # Cập nhật lịch sử huấn luyện
        training_history["episode_rewards"].append(episode_reward)
        training_history["episode_steps"].append(step)
        training_history["epsilon_values"].append(agent.epsilon)
        
        # Cập nhật tỷ lệ thành công (cửa sổ trượt 100 episode)
        success = 1 if info["status"] == "reached_goal" else 0
        success_window.append(success)
        if len(success_window) > 100:
            success_window.pop(0)
        training_history["success_rate"].append(sum(success_window) / len(success_window))
        
        # Hiển thị thông tin huấn luyện
        if verbose and episode % 10 == 0:
            avg_reward = np.mean(training_history["episode_rewards"][-10:])
            avg_steps = np.mean(training_history["episode_steps"][-10:])
            success_rate = sum(success_window) / len(success_window)
            
            print(f"Episode {episode}/{num_episodes} - "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Phần thưởng TB: {avg_reward:.1f}, "
                  f"Bước TB: {avg_steps:.1f}, "
                  f"Tỷ lệ thành công: {success_rate:.2f}")
        
        # Lưu mô hình định kỳ
        if save_path and save_interval > 0 and episode % save_interval == 0:
            save_file = os.path.join(save_path, f"{agent_type}_maze_{height}x{width}_ep{episode}.pkl")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            agent.save_model(save_file)
            if verbose:
                print(f"Đã lưu mô hình tại {save_file}")
    
    # Lưu mô hình cuối cùng
    if save_path:
        final_save_path = os.path.join(save_path, f"{agent_type}_maze_{height}x{width}_final.pkl")
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        agent.save_model(final_save_path)
        
        # Xuất ra định dạng JSON cho web (nếu cần)
        json_path = os.path.join(save_path, f"{agent_type}_maze_{height}x{width}_final.json")
        agent.export_to_json(json_path)
        
        if verbose:
            print(f"Đã lưu mô hình cuối cùng tại {final_save_path}")
            print(f"Đã xuất mô hình ra JSON tại {json_path}")
    
    # Thống kê thời gian huấn luyện
    training_time = time.time() - start_time
    if verbose:
        print(f"\nHoàn thành huấn luyện sau {training_time:.2f} giây.")
        print(f"Tỷ lệ thành công cuối cùng: {training_history['success_rate'][-1]:.2f}")
        print(f"Phần thưởng trung bình (100 episode cuối): "
              f"{np.mean(training_history['episode_rewards'][-100:]):.1f}")
    
    return training_history, agent

def evaluate_agent(agent: Any, maze_size: Tuple[int, int] = (10, 10),
                  num_episodes: int = 100, render: bool = False,
                  seed: Optional[int] = None) -> Dict[str, float]:
    """
    Đánh giá hiệu suất của agent đã huấn luyện.
    
    Args:
        agent (Any): Agent đã huấn luyện (QLearningAgent hoặc SARSAAgent)
        maze_size (Tuple[int, int]): Kích thước mê cung để đánh giá
        num_episodes (int): Số lượng episode đánh giá
        render (bool): Hiển thị quá trình đánh giá
        seed (int, optional): Hạt giống cho tính nhất quán
    
    Returns:
        Dict[str, float]: Kết quả đánh giá
    """
    height, width = maze_size
    
    # Tạo bộ sinh mê cung và môi trường mới để đánh giá
    maze_generator = DFSMazeGenerator(height=height, width=width, seed=seed)
    env = MazeEnvironment(maze_generator=maze_generator)
    
    # Tắt thăm dò trong quá trình đánh giá
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    # Các biến theo dõi
    rewards = []
    steps = []
    success_count = 0
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        if render:
            print(f"\nEpisode đánh giá {episode}:")
            env.render(mode='console')
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            step += 1
            
            if render:
                env.render(mode='console')
                print(f"Bước {step}: Hành động={info['action']}, "
                      f"Phần thưởng={reward:.1f}, Tổng phần thưởng={episode_reward:.1f}")
                time.sleep(0.1)
        
        # Cập nhật thống kê
        rewards.append(episode_reward)
        steps.append(step)
        if info["status"] == "reached_goal":
            success_count += 1
    
    # Khôi phục epsilon
    agent.epsilon = original_epsilon
    
    # Tính toán kết quả
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)
    success_rate = success_count / num_episodes
    
    results = {
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "success_rate": success_rate,
        "num_episodes": num_episodes
    }
    
    print(f"\nKết quả đánh giá trên mê cung {height}x{width}:")
    print(f"Phần thưởng trung bình: {avg_reward:.1f}")
    print(f"Số bước trung bình: {avg_steps:.1f}")
    print(f"Tỷ lệ thành công: {success_rate:.2f} ({success_count}/{num_episodes})")
    
    return results

def plot_training_data(training_history: Dict[str, List[float]], 
                      title_prefix: str = "", save_path: Optional[str] = None) -> None:
    """
    Vẽ biểu đồ tiến trình huấn luyện.
    
    Args:
        training_history (Dict[str, List[float]]): Lịch sử huấn luyện
        title_prefix (str): Tiền tố cho tiêu đề biểu đồ
        save_path (str, optional): Đường dẫn để lưu biểu đồ
    """
    plt.figure(figsize=(15, 10))
    
    # Biểu đồ phần thưởng
    plt.subplot(2, 2, 1)
    plt.plot(training_history["episode_rewards"])
    plt.title(f"{title_prefix} Phần thưởng theo episode")
    plt.xlabel("Episode")
    plt.ylabel("Tổng phần thưởng")
    plt.grid(True)
    
    # Biểu đồ số bước
    plt.subplot(2, 2, 2)
    plt.plot(training_history["episode_steps"])
    plt.title(f"{title_prefix} Số bước theo episode")
    plt.xlabel("Episode")
    plt.ylabel("Số bước")
    plt.grid(True)
    
    # Biểu đồ tỷ lệ thành công
    plt.subplot(2, 2, 3)
    plt.plot(training_history["success_rate"])
    plt.title(f"{title_prefix} Tỷ lệ thành công (cửa sổ trượt)")
    plt.xlabel("Episode")
    plt.ylabel("Tỷ lệ thành công")
    plt.grid(True)
    
    # Biểu đồ epsilon
    plt.subplot(2, 2, 4)
    plt.plot(training_history["epsilon_values"])
    plt.title(f"{title_prefix} Tỷ lệ thăm dò (epsilon)")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

if __name__ == "__main__":
    # Tạo thư mục để lưu mô hình
    models_dir = "backend/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Các tham số huấn luyện
    maze_size = (10, 10)  # Kích thước mê cung
    agent_type = "q_learning"  # Loại agent: "q_learning" hoặc "sarsa"
    num_episodes = 1000  # Số episode huấn luyện
    seed = 42  # Hạt giống
    
    # Đường dẫn lưu mô hình
    save_dir = os.path.join(models_dir, agent_type)
    
    # Huấn luyện agent
    training_history, agent = train_agent(
        maze_size=maze_size,
        agent_type=agent_type,
        num_episodes=num_episodes,
        save_path=save_dir,
        save_interval=200,  # Lưu mô hình sau mỗi 200 episode
        render_interval=0,  # 0 để tắt hiển thị, >0 để hiển thị mỗi n episode
        verbose=True,
        seed=seed
    )
    
    # Vẽ biểu đồ tiến trình huấn luyện
    plot_training_data(
        training_history,
        title_prefix=f"{agent_type.upper()} trên mê cung {maze_size[0]}x{maze_size[1]} - ",
        save_path=os.path.join(save_dir, f"{agent_type}_training_progress_{maze_size[0]}x{maze_size[1]}.png")
    )
    
    # Đánh giá agent sau khi huấn luyện
    evaluation_results = evaluate_agent(
        agent=agent,
        maze_size=maze_size,
        num_episodes=100,
        render=False,  # True để hiển thị quá trình đánh giá
        seed=seed+1  # Seed khác để đánh giá trên mê cung mới
    )