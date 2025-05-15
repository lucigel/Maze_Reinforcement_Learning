# training.py
import os
import numpy as np
import argparse
import time
from pathlib import Path

from maze_generators.dfs_generator import DFSMazeGenerator
from maze_generators.prim_generator import PrimMazeGenerator
from maze_generators.wilson_generator import WilsonMazeGenerator
from enviroment.maze_env import MazeEnvironment
from rl_agents.q_learning import QLearningAgent
from rl_agents.sarsa import SARSAAgent
from utils.config import *
from utils.data_handler import save_model
from utils.visualization import visualize_training_results

def parse_arguments():
    print(f"Parsing arguments for {PROJECT_NAME}")
    parser = argparse.ArgumentParser(description=f"{PROJECT_NAME} Training")
    parser.add_argument('--agent', type=str, default='q_learning', choices=['q_learning', 'sarsa'],
                        help='Thuật toán học tăng cường (q_learning, sarsa)')
    parser.add_argument('--maze', type=str, default='dfs', choices=['dfs', 'prim', 'wilson'],
                        help='Thuật toán sinh mê cung (dfs, prim, wilson)')
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large', 'xlarge'],
                        help='Kích thước mê cung (small, medium, large, xlarge)')
    parser.add_argument('--episodes', type=int, default=TRAINING_EPISODES,
                        help=f'Số episode huấn luyện (mặc định: {TRAINING_EPISODES})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Tốc độ học (mặc định: {LEARNING_RATE})')
    parser.add_argument('--gamma', type=float, default=DISCOUNT_FACTOR,
                        help=f'Hệ số giảm (mặc định: {DISCOUNT_FACTOR})')
    parser.add_argument('--epsilon', type=float, default=EXPLORATION_RATE,
                        help=f'Tỷ lệ khám phá ban đầu (mặc định: {EXPLORATION_RATE})')
    parser.add_argument('--decay', type=float, default=EXPLORATION_DECAY,
                        help=f'Tốc độ giảm tỷ lệ khám phá (mặc định: {EXPLORATION_DECAY})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Hạt giống ngẫu nhiên (mặc định: {RANDOM_SEED})')
    parser.add_argument('--render', action='store_true',
                        help='Hiển thị môi trường trong quá trình huấn luyện')
    parser.add_argument('--no-console', dest='console', action='store_false',
                        help='Tắt hiển thị trên console')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Thư mục lưu kết quả')
    
    return parser.parse_args()

def create_maze_generator(generator_type, maze_size):
    height, width = maze_size
    if generator_type == 'dfs':
        return DFSMazeGenerator(width=width, height=height)
    elif generator_type == 'prim':
        return PrimMazeGenerator(width=width, height=height)
    elif generator_type == 'wilson':
        return WilsonMazeGenerator(width=width, height=height)
    else:
        raise ValueError(f"Thuật toán sinh mê cung không hợp lệ: {generator_type}")

# training.py (chỉ cập nhật hàm create_agent)

def create_agent(agent_type, state_size, action_size, args):
    if agent_type == 'q_learning':
        return QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=args.lr,
            discount_factor=args.gamma,
            exploration_rate=args.epsilon,
            exploration_decay=args.decay,
            min_exploration_rate=MIN_EXPLORATION,
            seed=args.seed,
            use_double_q=USE_DOUBLE_Q  # Thêm tham số mới
        )
    elif agent_type == 'sarsa':
        return SARSAAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=args.lr,
            discount_factor=args.gamma,
            exploration_rate=args.epsilon,
            exploration_decay=args.decay,
            min_exploration_rate=MIN_EXPLORATION,
            seed=args.seed
        )
    else:
        raise ValueError(f"Thuật toán học tăng cường không hợp lệ: {agent_type}")

def save_model_with_history(agent, model_dir, agent_type, maze_size, episodes, episode_history=None):
    """Lưu mô hình và lịch sử huấn luyện"""
    # Tạo tên file
    filename = get_model_filename(agent_type, maze_size, episodes)
    
    # Tạo thư mục nếu chưa tồn tại
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Lưu mô hình
    save_path = os.path.join(model_dir, filename)
    agent.save_model(save_path)
    
    # Lưu lịch sử huấn luyện nếu có
    if episode_history:
        history_path = os.path.join(model_dir, f"{filename.replace('.pkl', '_history.npz')}")
        np.savez(history_path, **episode_history)
    
    return save_path

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Cấu hình dựa trên đối số
    maze_size = MAZE_SIZES[args.size]
    maze_generator = create_maze_generator(args.maze, maze_size)
    
    # Tạo môi trường mê cung
    env = MazeEnvironment(
        maze_generator=maze_generator,
        max_steps=MAX_STEPS,
        move_reward=MOVE_REWARD,
        wall_penalty=WALL_PENALTY,
        goal_reward=GOAL_REWARD,
        time_penalty=TIME_PENALTY
    )
    
    # Lấy kích thước không gian trạng thái và hành động
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    
    # Tạo agent
    agent = create_agent(args.agent, state_size, action_size, args)
    
    # Tạo thư mục lưu mô hình
    model_dir = Q_LEARNING_MODEL_DIR if args.agent == 'q_learning' else SARSA_MODEL_DIR
    
    print(f"\n{'=' * 50}")
    print(f"{PROJECT_NAME} - {VERSION}")
    print(f"{'=' * 50}")
    print(f"Thuật toán: {args.agent.upper()}")
    print(f"Bộ sinh mê cung: {args.maze.upper()}")
    print(f"Kích thước mê cung: {maze_size[0]}x{maze_size[1]} ({args.size})")
    print(f"Số episode huấn luyện: {args.episodes}")
    print(f"Tốc độ học (alpha): {args.lr}")
    print(f"Hệ số giảm (gamma): {args.gamma}")
    print(f"Tỷ lệ khám phá ban đầu (epsilon): {args.epsilon}")
    print(f"Tốc độ giảm tỷ lệ khám phá: {args.decay}")
    print(f"{'=' * 50}\n")
    
    # Hiển thị môi trường ban đầu
    if args.render:
        print("Môi trường mê cung ban đầu:")
        env.render(mode='console')
        print("\n")
    
    start_time = time.time()
    
    # Huấn luyện agent
    print(f"Bắt đầu huấn luyện {args.agent.upper()}...")
    training_results = agent.train(
        env=env,
        num_episodes=args.episodes,
        max_steps=MAX_STEPS,
        verbose=args.console,
        save_path=model_dir,
        save_interval=EVAL_INTERVAL
    )
    
    # Tính thời gian huấn luyện
    training_time = time.time() - start_time
    
    print(f"\nHoàn thành huấn luyện sau {training_time:.2f} giây!")
    
    # Lưu mô hình cuối cùng
    save_path = save_model_with_history(
        agent, 
        model_dir, 
        args.agent, 
        maze_size, 
        args.episodes, 
        training_results
    )
    print(f"Đã lưu mô hình cuối cùng tại: {save_path}")
    
    # Hiển thị kết quả huấn luyện
    visualize_training_results(
        training_results, 
        agent_type=args.agent, 
        maze_size=maze_size,
        save_dir=args.results_dir
    )
    
    # Hiển thị chính sách đã học
    print("\nChính sách đã học:")
    agent.visualize_policy(env.maze)
    
    # Hiển thị hàm giá trị
    print("\nHàm giá trị:")
    agent.visualize_value_function(env.maze)
    
    # Kiểm tra agent đã huấn luyện
    print("\nKiểm tra agent đã huấn luyện:")
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < MAX_STEPS:
        # Chọn hành động theo chính sách đã học (không khám phá)
        action = np.argmax(agent.q_table[state])
        
        # Thực hiện hành động
        next_state, reward, done, _ = env.step(action)
        
        # Cập nhật trạng thái và phần thưởng
        state = next_state
        total_reward += reward
        steps += 1
        
        # Hiển thị môi trường nếu cần
        if args.render and steps % 5 == 0:  # Hiển thị mỗi 5 bước
            env.render(mode='console')
            print(f"Bước: {steps}, Phần thưởng: {reward:.2f}, Tổng phần thưởng: {total_reward:.2f}")
            time.sleep(0.1)  # Chờ để có thể xem
    
    # Hiển thị kết quả kiểm tra
    print(f"\nKết quả kiểm tra:")
    print(f"Số bước: {steps}")
    print(f"Tổng phần thưởng: {total_reward:.2f}")
    print(f"Trạng thái hoàn thành: {'Đạt đích' if done and steps < MAX_STEPS else 'Không đạt'}")
    
    # So sánh với đường đi ngắn nhất (nếu có)
    shortest_path = env.get_shortest_path()
    if shortest_path:
        print(f"Độ dài đường đi ngắn nhất: {len(shortest_path) - 1}")
        print(f"Hiệu suất agent: {(len(shortest_path) - 1) / steps:.2%}")
    
    print("\nQuá trình huấn luyện và kiểm tra hoàn tất!")

if __name__ == "__main__":
    main()