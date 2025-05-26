# evaluation.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import time
import json
from tqdm import tqdm
import torch

from rl_agents.q_learning import QLearningAgent
from rl_agents.sarsa import SARSAAgent
from rl_agents.dqn_agent import DQNAgent
from enviroment.maze_env import MazeEnvironment
from maze_generators.dfs_generator import DFSMazeGenerator
from maze_generators.prim_generator import PrimMazeGenerator
from maze_generators.wilson_generator import WilsonMazeGenerator
from utils.data_handler import load_model, save_training_history, load_training_history
from utils.visualization import visualize_maze, visualize_training_results, visualize_heatmap, visualize_maze_with_path


def parse_arguments():
    """
    Phân tích tham số dòng lệnh
    """
    parser = argparse.ArgumentParser(description='Đánh giá hiệu suất các mô hình học tăng cường trên mê cung')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Đường dẫn đến mô hình đã huấn luyện')
    
    parser.add_argument('--model_type', type=str, choices=['q_learning', 'sarsa', 'dqn'],
                        default='q_learning', help='Loại mô hình cần đánh giá')
    
    parser.add_argument('--maze_type', type=str, choices=['dfs', 'prim', 'wilson'],
                        default='dfs', help='Loại mê cung để đánh giá')
    
    parser.add_argument('--maze_size', type=int, default=10,
                        help='Kích thước mê cung (NxN)')
    
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Số episode đánh giá')
    
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Số bước tối đa trong mỗi episode')
    
    parser.add_argument('--render', action='store_true',
                        help='Hiển thị mê cung trong quá trình đánh giá')
    
    parser.add_argument('--render_delay', type=float, default=0.1,
                        help='Độ trễ giữa các bước khi hiển thị (giây)')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Thư mục lưu kết quả đánh giá')
    
    parser.add_argument('--compare', action='store_true',
                        help='So sánh với mô hình khác')
    
    parser.add_argument('--compare_path', type=str,
                        help='Đường dẫn đến mô hình so sánh')
    
    parser.add_argument('--compare_type', type=str, choices=['q_learning', 'sarsa', 'dqn'],
                        help='Loại mô hình so sánh')
    
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='Mức độ nhiễu trong môi trường (0.0-1.0)')
    
    parser.add_argument('--save_video', action='store_true',
                        help='Lưu video quá trình đánh giá')
    
    return parser.parse_args()


def create_maze_generator(maze_type: str, maze_size: int) -> Any:
    """
    Tạo bộ sinh mê cung dựa trên loại được chỉ định
    """
    if maze_type == 'dfs':
        return DFSMazeGenerator(maze_size, maze_size)
    elif maze_type == 'prim':
        return PrimMazeGenerator(maze_size, maze_size)
    elif maze_type == 'wilson':
        return WilsonMazeGenerator(maze_size, maze_size)
    else:
        raise ValueError(f"Loại mê cung không hợp lệ: {maze_type}")


def create_agent(model_type: str, state_size: Tuple[int, int], action_size: int) -> Any:
    """
    Tạo agent dựa trên loại được chỉ định
    """
    if model_type == 'q_learning':
        return QLearningAgent(state_size, action_size)
    elif model_type == 'sarsa':
        return SARSAAgent(state_size, action_size)
    elif model_type == 'dqn':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return DQNAgent(
            state_size=state_size,
            action_size=action_size,
            device=device
        )
    else:
        raise ValueError(f"Loại mô hình không hợp lệ: {model_type}")


def evaluate_model(agent: Any, env: MazeEnvironment, num_episodes: int, max_steps: int,
                   render: bool = False, render_delay: float = 0.1, noise_level: float = 0.0,
                   model_type: str = None) -> Dict[str, Any]:
    """
    Đánh giá hiệu suất của mô hình
    """
    episode_rewards = []
    episode_steps = []
    success_count = 0
    all_paths = []
    
    # Đặt epsilon = 0 để agent luôn khai thác kiến thức đã học
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    # Nếu là DQN, set current_maze
    if model_type == 'dqn' and hasattr(agent, 'current_maze'):
        agent.current_maze = env.maze
    
    for episode in tqdm(range(num_episodes), desc="Evaluating episodes"):
        state = env.reset()
        episode_reward = 0
        steps = 0
        path = [state]
        
        done = False
        while not done and steps < max_steps:
            # Chọn hành động
            if np.random.random() < noise_level:
                action = np.random.randint(0, env.get_action_size())
            else:
                if model_type == 'dqn':
                    action = agent.choose_action(state, env.maze)
                else:
                    action = agent.choose_action(state)
            
            # Thực hiện hành động
            next_state, reward, done, info = env.step(action)
            
            # Cập nhật thông tin
            state = next_state
            episode_reward += reward
            steps += 1
            path.append(state)
            
            # Hiển thị nếu được yêu cầu
            if render:
                env.render(mode='console')
                time.sleep(render_delay)
        
        # Kiểm tra thành công
        if done:
            success_count += 1
        
        # Lưu thông tin episode
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        all_paths.append(path)
    
    # Khôi phục epsilon
    agent.epsilon = original_epsilon
    
    # Tính tỷ lệ thành công
    success_rate = (success_count / num_episodes) * 100
    
    # Tổng hợp kết quả
    results = {
        'rewards': episode_rewards,
        'steps': episode_steps,
        'success_rate': success_rate,
        'avg_reward': np.mean(episode_rewards),
        'avg_steps': np.mean(episode_steps),
        'min_steps': np.min(episode_steps) if episode_steps else 0,
        'max_steps': np.max(episode_steps) if episode_steps else 0,
        'std_steps': np.std(episode_steps) if episode_steps else 0,
        'paths': all_paths
    }
    
    return results


def visualize_results(env: MazeEnvironment, agent: Any, results: Dict[str, Any], output_dir: str, model_type: str = None) -> None:
    """
    Hiển thị kết quả đánh giá
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Nếu là DQN, set current_maze
    if model_type == 'dqn' and hasattr(agent, 'current_maze'):
        agent.current_maze = env.maze
    
    # 1. Hiển thị chính sách và giá trị
    policy = agent.get_policy()
    value_function = agent.get_value_function()
    
    plt.figure(figsize=(15, 12))
    
    # Vẽ mê cung
    plt.subplot(2, 2, 1)
    maze_array = env.maze.copy()
    plt.imshow(maze_array, cmap='binary')
    plt.title('Maze Structure')
    plt.colorbar(ticks=[0, 1, 2, 3], label='Cell Type')
    
    # Vẽ heatmap giá trị
    plt.subplot(2, 2, 2)
    masked_values = np.ma.masked_where(env.maze == 1, value_function)
    plt.imshow(masked_values, cmap='hot')
    plt.title('Value Function')
    plt.colorbar()
    
    # Vẽ biểu đồ phân bố số bước
    plt.subplot(2, 2, 3)
    plt.hist(results['steps'], bins=20, color='blue', alpha=0.7)
    plt.axvline(results['avg_steps'], color='red', linestyle='--', label=f'Avg: {results["avg_steps"]:.2f}')
    plt.title('Step Distribution')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Vẽ biểu đồ phân bố phần thưởng
    plt.subplot(2, 2, 4)
    plt.hist(results['rewards'], bins=20, color='green', alpha=0.7)
    plt.axvline(results['avg_reward'], color='red', linestyle='--', label=f'Avg: {results["avg_reward"]:.2f}')
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_summary.png'), dpi=300)
    plt.close()
    
    # 2. Hiển thị đường đi tốt nhất
    if results['rewards']:
        best_episode_idx = np.argmax(results['rewards'])
        best_path = results['paths'][best_episode_idx]
        
        plt.figure(figsize=(10, 10))
        visualize_maze_with_path(env.maze, best_path)
        plt.title(f'Best Path (Reward: {results["rewards"][best_episode_idx]:.2f}, Steps: {results["steps"][best_episode_idx]})')
        plt.savefig(os.path.join(output_dir, 'best_path.png'), dpi=300)
        plt.close()
    
    # 3. Hiển thị heatmap các ô đã thăm
    plt.figure(figsize=(10, 10))
    visualize_heatmap(agent.state_visits, env.maze)
    plt.title('State Visitation Heatmap')
    plt.savefig(os.path.join(output_dir, 'state_visitation.png'), dpi=300)
    plt.close()
    
    # Hiển thị thông tin tổng hợp
    print("\n===== EVALUATION RESULTS =====")
    print(f"Success Rate: {results['success_rate']:.2f}%")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Average Steps: {results['avg_steps']:.2f}")
    print(f"Minimum Steps: {results['min_steps']}")
    print(f"Maximum Steps: {results['max_steps']}")
    print(f"Standard Deviation of Steps: {results['std_steps']:.2f}")
    print("=============================\n")


def compare_models(agent1: Any, agent2: Any, env: MazeEnvironment, num_episodes: int, max_steps: int,
                  output_dir: str, model1_type: str = None, model2_type: str = None) -> None:
    """
    So sánh hiệu suất của hai mô hình
    """
    # Đánh giá từng mô hình
    print("\nEvaluating first model...")
    results1 = evaluate_model(agent1, env, num_episodes, max_steps, model_type=model1_type)
    
    print("\nEvaluating second model...")
    env.reset()
    results2 = evaluate_model(agent2, env, num_episodes, max_steps, model_type=model2_type)
    
    # So sánh kết quả
    plt.figure(figsize=(15, 12))
    
    # So sánh phân bố số bước
    plt.subplot(2, 2, 1)
    plt.hist(results1['steps'], bins=20, alpha=0.5, label=f'{model1_type or "Model 1"}')
    plt.hist(results2['steps'], bins=20, alpha=0.5, label=f'{model2_type or "Model 2"}')
    plt.axvline(results1['avg_steps'], color='blue', linestyle='--', label=f'{model1_type or "Model 1"} Avg: {results1["avg_steps"]:.2f}')
    plt.axvline(results2['avg_steps'], color='orange', linestyle='--', label=f'{model2_type or "Model 2"} Avg: {results2["avg_steps"]:.2f}')
    plt.title('Steps Comparison')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.legend()
    
    # So sánh phân bố phần thưởng
    plt.subplot(2, 2, 2)
    plt.hist(results1['rewards'], bins=20, alpha=0.5, label=f'{model1_type or "Model 1"}')
    plt.hist(results2['rewards'], bins=20, alpha=0.5, label=f'{model2_type or "Model 2"}')
    plt.axvline(results1['avg_reward'], color='blue', linestyle='--', label=f'{model1_type or "Model 1"} Avg: {results1["avg_reward"]:.2f}')
    plt.axvline(results2['avg_reward'], color='orange', linestyle='--', label=f'{model2_type or "Model 2"} Avg: {results2["avg_reward"]:.2f}')
    plt.title('Rewards Comparison')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    
    # So sánh giá trị Q
    plt.subplot(2, 2, 3)
    value_diff = agent2.get_value_function() - agent1.get_value_function()
    im = plt.imshow(value_diff, cmap='coolwarm')
    plt.title(f'Value Function Difference ({model2_type or "Model 2"} - {model1_type or "Model 1"})')
    plt.colorbar(im)
    
    # So sánh chính sách
    plt.subplot(2, 2, 4)
    policy1 = agent1.get_policy()
    policy2 = agent2.get_policy()
    policy_diff = (policy1 != policy2).astype(int)
    plt.imshow(policy_diff, cmap='gray')
    plt.title('Policy Differences')
    plt.colorbar(ticks=[0, 1], label='Same/Different')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    # Hiển thị thông tin tổng hợp
    print("\n===== MODEL COMPARISON =====")
    print(f"{model1_type or 'Model 1'} Success Rate: {results1['success_rate']:.2f}%")
    print(f"{model2_type or 'Model 2'} Success Rate: {results2['success_rate']:.2f}%")
    print(f"{model1_type or 'Model 1'} Average Reward: {results1['avg_reward']:.2f}")
    print(f"{model2_type or 'Model 2'} Average Reward: {results2['avg_reward']:.2f}")
    print(f"{model1_type or 'Model 1'} Average Steps: {results1['avg_steps']:.2f}")
    print(f"{model2_type or 'Model 2'} Average Steps: {results2['avg_steps']:.2f}")
    
    # Tính tỷ lệ sự khác biệt về chính sách
    policy_diff_rate = np.sum(policy_diff) / (policy_diff.shape[0] * policy_diff.shape[1]) * 100
    print(f"Policy Difference Rate: {policy_diff_rate:.2f}%")
    print("=============================\n")


def test_robustness(agent: Any, maze_type: str, maze_sizes: List[int], num_episodes: int, 
                    max_steps: int, output_dir: str, model_type: str = None) -> Dict[str, List[float]]:
    """
    Kiểm tra khả năng tổng quát hóa của mô hình trên các mê cung kích thước khác nhau
    """
    success_rates = []
    avg_rewards = []
    avg_steps = []
    
    for size in maze_sizes:
        print(f"\nTesting on maze size {size}x{size}...")
        
        # Tạo mê cung mới
        maze_generator = create_maze_generator(maze_type, size)
        env = MazeEnvironment(maze_generator=maze_generator)
        
        # Đánh giá trên mê cung mới
        results = evaluate_model(agent, env, num_episodes // len(maze_sizes), max_steps, model_type=model_type)
        
        # Lưu kết quả
        success_rates.append(results['success_rate'])
        avg_rewards.append(results['avg_reward'])
        avg_steps.append(results['avg_steps'])
    
    # Vẽ biểu đồ kết quả
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(maze_sizes, success_rates, 'o-', color='blue', markersize=8)
    plt.title('Success Rate vs Maze Size')
    plt.xlabel('Maze Size')
    plt.ylabel('Success Rate (%)')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(maze_sizes, avg_rewards, 'o-', color='green', markersize=8)
    plt.title('Average Reward vs Maze Size')
    plt.xlabel('Maze Size')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(maze_sizes, avg_steps, 'o-', color='red', markersize=8)
    plt.title('Average Steps vs Maze Size')
    plt.xlabel('Maze Size')
    plt.ylabel('Average Steps')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_test.png'), dpi=300)
    plt.close()
    
    # Lưu kết quả
    robustness_results = {
        'maze_sizes': maze_sizes,
        'success_rates': success_rates,
        'avg_rewards': avg_rewards,
        'avg_steps': avg_steps
    }
    
    with open(os.path.join(output_dir, 'robustness_results.json'), 'w') as f:
        json.dump(robustness_results, f, indent=4)
    
    return robustness_results


def analyze_path_efficiency(env: MazeEnvironment, agent: Any, num_episodes: int, max_steps: int,
                           output_dir: str, model_type: str = None) -> None:
    """
    Phân tích hiệu quả đường đi của agent
    """
    # Tính đường đi ngắn nhất
    optimal_path = env.get_shortest_path()
    optimal_length = len(optimal_path) - 1 if optimal_path else 0
    
    if optimal_length == 0:
        print("Cannot find optimal path!")
        return
    
    # Đánh giá đường đi của agent
    results = evaluate_model(agent, env, num_episodes, max_steps, model_type=model_type)
    
    # Tính tỷ lệ hiệu quả
    efficiency_ratios = []
    for steps in results['steps']:
        if steps > 0 and steps < max_steps:  # Chỉ tính cho các episode thành công
            efficiency_ratios.append(optimal_length / steps)
    
    if not efficiency_ratios:
        print("No successful episodes to analyze!")
        return
    
    # Vẽ biểu đồ phân phối hiệu quả
    plt.figure(figsize=(10, 6))
    plt.hist(efficiency_ratios, bins=20, color='purple', alpha=0.7)
    plt.axvline(np.mean(efficiency_ratios), color='red', linestyle='--', 
                label=f'Avg Efficiency: {np.mean(efficiency_ratios):.2f}')
    plt.title('Path Efficiency Distribution')
    plt.xlabel('Efficiency Ratio (Optimal Steps / Agent Steps)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'path_efficiency.png'), dpi=300)
    plt.close()
    
    # Hiển thị thông tin tổng hợp
    print("\n===== PATH EFFICIENCY ANALYSIS =====")
    print(f"Optimal Path Length: {optimal_length}")
    print(f"Average Agent Path Length: {results['avg_steps']:.2f}")
    print(f"Average Efficiency Ratio: {np.mean(efficiency_ratios):.2f}")
    print(f"Maximum Efficiency Ratio: {np.max(efficiency_ratios):.2f}")
    print(f"Minimum Efficiency Ratio: {np.min(efficiency_ratios):.2f}")
    print("====================================\n")


def main():
    """
    Hàm chính thực hiện đánh giá
    """
    # Phân tích tham số dòng lệnh
    args = parse_arguments()
    
    # Tạo thư mục đầu ra
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tạo môi trường mê cung
    maze_generator = create_maze_generator(args.maze_type, args.maze_size)
    env = MazeEnvironment(maze_generator=maze_generator, max_steps=args.max_steps)
    
    # Tạo agent
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    agent = create_agent(args.model_type, state_size, action_size)
    
    # Tải mô hình
    try:
        if args.model_type == 'dqn':
            agent.load_model(args.model_path)
            agent.current_maze = env.maze
        else:
            load_model(agent, args.model_path)
        print(f"Loaded {args.model_type} model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Đánh giá mô hình
    print(f"\nEvaluating {args.model_type} model on {args.maze_type} maze of size {args.maze_size}x{args.maze_size}...")
    results = evaluate_model(
        agent, env, args.num_episodes, args.max_steps, 
        render=args.render, render_delay=args.render_delay,
        noise_level=args.noise_level,
        model_type=args.model_type
    )
    
    # Hiển thị kết quả
    visualize_results(env, agent, results, args.output_dir, model_type=args.model_type)
    
    # Phân tích hiệu quả đường đi
    analyze_path_efficiency(env, agent, args.num_episodes, args.max_steps, args.output_dir, model_type=args.model_type)
    
    # So sánh với mô hình khác nếu được yêu cầu
    if args.compare and args.compare_path:
        # Tạo agent so sánh
        compare_agent = create_agent(args.compare_type or args.model_type, state_size, action_size)
        
        # Tải mô hình so sánh
        try:
            if args.compare_type == 'dqn':
                compare_agent.load_model(args.compare_path)
                compare_agent.current_maze = env.maze
            else:
                load_model(compare_agent, args.compare_path)
            print(f"Loaded comparison model from {args.compare_path}")
            
            # So sánh hai mô hình
            compare_models(agent, compare_agent, env, args.num_episodes, args.max_steps, 
                         args.output_dir, model1_type=args.model_type, model2_type=args.compare_type)
        except Exception as e:
            print(f"Error loading comparison model: {e}")
    
    # Kiểm tra khả năng tổng quát hóa
    maze_sizes = [5, 10, 15, 20]
    print("\nTesting robustness across different maze sizes...")
    test_robustness(agent, args.maze_type, maze_sizes, args.num_episodes, args.max_steps, 
                   args.output_dir, model_type=args.model_type)
    
    print(f"\nEvaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()