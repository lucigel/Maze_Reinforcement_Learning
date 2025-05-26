# training.py 
import os
import numpy as np
import argparse
import time
from pathlib import Path
import torch

from maze_generators.dfs_generator import DFSMazeGenerator
from maze_generators.prim_generator import PrimMazeGenerator
from maze_generators.wilson_generator import WilsonMazeGenerator
from enviroment.maze_env import MazeEnvironment
from rl_agents.q_learning import QLearningAgent
from rl_agents.sarsa import SARSAAgent
from rl_agents.dqn_agent import DQNAgent 
from utils.config import *
from utils.visualization import visualize_training_results


def parse_arguments():
    print(f"Parsing arguments for {PROJECT_NAME}")
    parser = argparse.ArgumentParser(description=f"{PROJECT_NAME} Training")
    parser.add_argument('--agent', type=str, default='q_learning', choices=['q_learning', 'sarsa', 'dqn'],
                        help='Thuật toán học tăng cường (q_learning, sarsa, dqn)')
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
    parser.add_argument('--hidden-size', type=int, default=DQN_HIDDEN_SIZE,
                        help=f'Kích thước lớp ẩn cho DQN (mặc định: {DQN_HIDDEN_SIZE})')
    parser.add_argument('--batch-size', type=int, default=DQN_BATCH_SIZE,
                        help=f'Kích thước batch cho DQN (mặc định: {DQN_BATCH_SIZE})')
    parser.add_argument('--buffer-size', type=int, default=DQN_BUFFER_SIZE,
                        help=f'Kích thước buffer cho DQN (mặc định: {DQN_BUFFER_SIZE})')
    parser.add_argument('--target-update', type=int, default=DQN_TARGET_UPDATE_FREQ,
                        help=f'Tần suất cập nhật target network (mặc định: {DQN_TARGET_UPDATE_FREQ})')
    
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
            use_double_q=USE_DOUBLE_Q
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
    elif agent_type == 'dqn':
        # Tạo DQN agent với các tham số phù hợp
        return DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=args.lr if args.lr != LEARNING_RATE else DQN_LEARNING_RATE,  # Sử dụng LR thích hợp cho DQN
            discount_factor=args.gamma,
            exploration_rate=args.epsilon,
            exploration_decay=args.decay,
            min_exploration_rate=MIN_EXPLORATION,
            seed=args.seed,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            target_update_freq=args.target_update,
            hidden_size=args.hidden_size,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        raise ValueError(f"Thuật toán học tăng cường không hợp lệ: {agent_type}")

def save_model_with_history(agent, model_dir, agent_type, maze_size, episodes, episode_history=None):
    """Lưu mô hình và lịch sử huấn luyện"""
    
    # Tạo thư mục nếu chưa tồn tại
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Tạo tên file dựa trên agent_type, maze_size và episodes
    height, width = maze_size
    
    if agent_type == 'dqn':
        # Lưu file .pth cho DQN
        filename = f"dqn_{height}x{width}_{episodes}ep.pth"
        save_path = os.path.join(model_dir, filename)
        agent.save_model(save_path)
        
        # DQN agent đã tự động lưu .pkl trong hàm save_model()
        # nên không cần lưu thêm
    else:
        # Lưu file .pkl cho Q-Learning và SARSA
        filename = f"{agent_type}_{height}x{width}_{episodes}ep.pkl"
        save_path = os.path.join(model_dir, filename)
        agent.save_model(save_path)
    
    # Lưu history nếu có
    if episode_history:
        history_filename = f"{agent_type}_{height}x{width}_{episodes}ep_history.npz"
        history_path = os.path.join(model_dir, history_filename)
        np.savez(history_path, **episode_history)
    
    return save_path

def get_model_filename(agent_type, maze_size, episodes):
    """Tạo tên file mô hình dựa trên thông số"""
    height, width = maze_size
    size_str = f"{height}x{width}"
    
    if agent_type == 'dqn':
        return f"dqn_maze_{size_str}_e{episodes}.pth"
    else:
        return f"{agent_type}_maze_{size_str}_e{episodes}.pkl"

def main():
    args = parse_arguments()

    np.random.seed(args.seed)
    if args.agent == 'dqn':
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    maze_size = MAZE_SIZES[args.size]
    maze_generator = create_maze_generator(args.maze, maze_size)
    
    env = MazeEnvironment(
        maze_generator=maze_generator,
        max_steps=MAX_STEPS,
        move_reward=MOVE_REWARD,
        wall_penalty=WALL_PENALTY,
        goal_reward=GOAL_REWARD,
        time_penalty=TIME_PENALTY
    )
    
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    
    agent = create_agent(args.agent, state_size, action_size, args)
    
    if args.agent == 'q_learning':
        model_dir = Q_LEARNING_MODEL_DIR
    elif args.agent == 'sarsa':
        model_dir = SARSA_MODEL_DIR
    else:  
        model_dir = DQN_MODEL_DIR
    
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
    
    if args.agent == 'dqn':
        print(f"Kích thước lớp ẩn: {args.hidden_size}")
        print(f"Kích thước batch: {args.batch_size}")
        print(f"Kích thước buffer: {args.buffer_size}")
        print(f"Tần suất cập nhật target network: {args.target_update}")
        print(f"Thiết bị: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    print(f"{'=' * 50}\n")
    
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
    
    training_time = time.time() - start_time
    
    print(f"\nHoàn thành huấn luyện sau {training_time:.2f} giây!")
    
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
    # Thêm tham số loss_key cho DQN vì nó trả về thêm thông tin loss
    loss_key = 'losses' if args.agent == 'dqn' else None
    visualize_training_results(
        training_results, 
        agent_type=args.agent, 
        maze_size=maze_size,
        save_dir=args.results_dir,
        loss_key=loss_key  # Thêm tham số này
    )
    
    print("\nChính sách đã học:")
    agent.visualize_policy(env.maze)
    
    print("\nHàm giá trị:")
    agent.visualize_value_function(env.maze)
    
    # Kiểm tra agent đã huấn luyện
    print("\nKiểm tra agent đã huấn luyện:")
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    # Đặt epsilon = 0 để đảm bảo không có khám phá ngẫu nhiên
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    while not done and steps < MAX_STEPS:
        # Chọn hành động theo chính sách đã học (không khám phá)
        if args.agent == 'dqn':
            # DQN cần mê cung làm input
            action = agent.choose_action(state)
        else:
            # Q-Learning và SARSA sử dụng Q-table trực tiếp
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
    
  
    agent.epsilon = original_epsilon
    
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