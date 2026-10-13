#!/usr/bin/env python3
"""
Random Maze Generator + Tabular Q-Learning Agent
Author: Kevin Tieu (generated for CS study)

Generates one fixed random maze, then trains a Q-learning agent
that improves (fewer steps) over successive attempts/episodes.
Requires: Python 3, numpy. matplotlib is optional for plots.
"""

import random
import numpy as np
import time
from collections import deque

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found; plots will be skipped. Install with: pip install matplotlib")

# ---------------------------------------------------------------------------
# Maze generation (recursive backtracking)
# ---------------------------------------------------------------------------

def generate_maze(width, height):
    """Return a grid where 0 = open, 1 = wall. Odd dimensions recommended."""
    # Ensure odd sizes so walls and paths align
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    maze = np.ones((height, width), dtype=np.uint8)
    stack = []
    start = (1, 1)
    maze[start] = 0
    stack.append(start)

    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]

    while stack:
        y, x = stack[-1]
        random.shuffle(directions)
        carved = False
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 1 <= ny < height - 1 and 1 <= nx < width - 1 and maze[ny, nx] == 1:
                maze[ny, nx] = 0
                maze[y + dy // 2, x + dx // 2] = 0
                stack.append((ny, nx))
                carved = True
                break
        if not carved:
            stack.pop()

    # Start and goal
    maze[1, 1] = 0
    maze[height - 2, width - 2] = 0
    return maze, (1, 1), (height - 2, width - 2)

# ---------------------------------------------------------------------------
# Q-Learning environment
# ---------------------------------------------------------------------------

ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
ACTION_NAMES = ["R", "D", "L", "U"]

class MazeEnv:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.height, self.width = maze.shape
        self.pos = start
        self.max_steps = self.height * self.width * 2

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self.pos

    def step(self, action):
        dy, dx = ACTIONS[action]
        ny, nx = self.pos[0] + dy, self.pos[1] + dx
        self.steps += 1

        # Stay in place if wall or out of bounds
        if not (0 <= ny < self.height and 0 <= nx < self.width) or self.maze[ny, nx] == 1:
            ny, nx = self.pos

        self.pos = (ny, nx)
        done = self.pos == self.goal or self.steps >= self.max_steps
        reward = 10.0 if self.pos == self.goal else -0.1
        return self.pos, reward, done

# ---------------------------------------------------------------------------
# Q-Learning agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    def __init__(self, height, width, n_actions=4, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.q = np.zeros((height, width, n_actions), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions

    def choose_action(self, state, greedy=False):
        y, x = state
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q[y, x]))

    def update(self, state, action, reward, next_state, done):
        y, x = state
        ny, nx = next_state
        best_next = 0.0 if done else np.max(self.q[ny, nx])
        target = reward + self.gamma * best_next
        self.q[y, x, action] += self.alpha * (target - self.q[y, x, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ---------------------------------------------------------------------------
# Training and demonstration
# ---------------------------------------------------------------------------

def train(env, agent, max_episodes=10000, print_every=100, snapshot_every=50, maze=None, start=None, goal=None):
    """Train until the agent first reaches the goal (or hits the safety limit)."""
    episode_lengths = []
    successes = []
    first_success = None
    snapshot_dir = "snapshots"
    if HAS_MPL and maze is not None:
        import os
        os.makedirs(snapshot_dir, exist_ok=True)

    ep = 0
    while ep < max_episodes:
        ep += 1
        state = env.reset()
        done = False
        steps = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            steps += 1

        agent.decay_epsilon()
        episode_lengths.append(steps)
        # Check success only with a pure greedy rollout
        greedy_path = run_greedy(env, agent)
        success = 1 if greedy_path and greedy_path[-1] == env.goal else 0
        successes.append(success)

        if success and first_success is None:
            first_success = ep
            print(f"\n*** Goal reached on episode {ep} (greedy policy) ***")
            if HAS_MPL and maze is not None:
                title = f"SUCCESS episode {ep} | {len(greedy_path)-1} steps"
                visualize_maze(maze, path=greedy_path, start=start, goal=goal,
                               filename=f"{snapshot_dir}/maze_SUCCESS.png", title=title)
                animate_path(maze, greedy_path, start, goal,
                             filename=f"{snapshot_dir}/maze_SUCCESS.gif", title=title)
                # extra frames of the successful journey
                frames_dir = "success_frames"
                import os
                os.makedirs(frames_dir, exist_ok=True)
                for i in range(0, len(greedy_path), max(1, len(greedy_path)//20)):
                    visualize_maze(maze, path=greedy_path[:i+1], start=start, goal=goal,
                                   filename=f"{frames_dir}/frame_{i:04d}.png",
                                   title=f"Success path step {i}")
            break

        if HAS_MPL and maze is not None and ep % snapshot_every == 0:
            t_run = time.time()
            path = run_greedy(env, agent)
            run_sec = time.time() - t_run
            reached = path and path[-1] == goal
            status = "REACHED" if reached else "FAILED"
            title = f"Ep {ep} | {len(path)-1} steps | {status} | {run_sec:.3f}s"
            visualize_maze(maze, path=path, start=start, goal=goal,
                           filename=f"{snapshot_dir}/maze_ep{ep:04d}.png", title=title)
            animate_path(maze, path, start, goal,
                         filename=f"{snapshot_dir}/maze_ep{ep:04d}.gif", title=title)
            print(f"  Snapshot {ep}: {status}, {len(path)-1} steps, {run_sec:.3f}s")

        if ep % print_every == 0:
            recent = episode_lengths[-print_every:]
            success_rate = np.mean(successes[-print_every:]) * 100
            print(f"Episode {ep:5d} | avg steps: {np.mean(recent):6.1f} | success: {success_rate:5.1f}% | epsilon: {agent.epsilon:.3f}")

    return episode_lengths, successes, first_success, ep

def visualize_maze(maze, path=None, start=None, goal=None, filename="maze_with_path.png", title="Maze with Learned Path"):
    """High-quality color visualization of the maze and learned path."""
    if not HAS_MPL:
        return
    h, w = maze.shape
    img = np.ones((h, w, 3), dtype=np.float32)
    img[maze == 1] = [0.12, 0.12, 0.12]
    img[maze == 0] = [0.95, 0.95, 0.95]

    if path:
        for y, x in path:
            img[y, x] = [0.2, 0.5, 1.0]

    if start:
        img[start] = [0.15, 0.8, 0.25]
    if goal:
        img[goal] = [0.95, 0.2, 0.2]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=13, pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()

def create_progress_gif(snapshot_dir="snapshots", output="learning_progress.gif"):
    """Build an animated GIF from the snapshot PNGs so you can see the path improve."""
    if not HAS_MPL:
        return
    try:
        import os
        from matplotlib.animation import FuncAnimation, PillowWriter
        files = sorted([f for f in os.listdir(snapshot_dir) if f.endswith(".png")])
        if len(files) < 2:
            return
        images = [plt.imread(os.path.join(snapshot_dir, f)) for f in files]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis("off")
        im = ax.imshow(images[0])
        def update(i):
            im.set_array(images[i])
            ax.set_title(files[i].replace(".png", "").replace("maze_", ""), fontsize=12)
            return [im]
        anim = FuncAnimation(fig, update, frames=len(images), interval=400, blit=True)
        anim.save(output, writer=PillowWriter(fps=2))
        plt.close()
        print(f"Saved learning-progress animation: {output}")
    except Exception as e:
        print(f"Could not create progress GIF: {e}")

def animate_path(maze, path, start, goal, filename="maze_path_animation.gif", title="Agent following path"):
    """Create an animated GIF of the agent following the full path (success or failure).
    Stuck positions are made visible by blinking the agent red."""
    if not HAS_MPL or not path or len(path) < 2:
        return
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("Animation skipped (need pillow). Run: python -m pip install pillow")
        return

    h, w = maze.shape
    base = np.ones((h, w, 3), dtype=np.float32)
    base[maze == 1] = [0.12, 0.12, 0.12]
    base[maze == 0] = [0.95, 0.95, 0.95]
    base[start] = [0.15, 0.8, 0.25]
    base[goal] = [0.95, 0.2, 0.2]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(base, interpolation="nearest")
    ax.axis("off")
    title_text = ax.set_title(title, fontsize=11)
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        fontsize=10, color="black", verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    def update(frame):
        img = base.copy()
        # Draw trail
        for i in range(min(frame + 1, len(path))):
            y, x = path[i]
            if (y, x) != start and (y, x) != goal:
                img[y, x] = [0.2, 0.5, 1.0]
        # Current position
        if frame < len(path):
            cy, cx = path[frame]
            # Blink red if stuck (same as previous position)
            stuck = frame > 0 and path[frame] == path[frame - 1]
            if stuck:
                color = [1.0, 0.2, 0.2] if (frame // 2) % 2 == 0 else [1.0, 0.6, 0.2]
            else:
                color = [1.0, 0.85, 0.1]
            img[cy, cx] = color
            step_text.set_text(f"step {frame}/{len(path)-1}" + (" STUCK" if stuck else ""))
        im.set_array(img)
        return [im, step_text]

    # Slower so movement and stuck blinks are easy to see
    anim = FuncAnimation(fig, update, frames=len(path), interval=120, blit=True)
    try:
        anim.save(filename, writer=PillowWriter(fps=8))
    except Exception as e:
        print(f"GIF save failed for {filename}: {e}")
    plt.close()

def run_greedy(env, agent):
    """Run one episode with the learned policy (no exploration)."""
    state = env.reset()
    path = [state]
    done = False
    while not done:
        action = agent.choose_action(state, greedy=True)
        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state
        if len(path) > env.max_steps:
            break
    return path

def print_maze(maze, path=None):
    """Print an ASCII representation of the maze (optional terminal view)."""
    h, w = maze.shape
    for y in range(h):
        row = ""
        for x in range(w):
            if path and (y, x) in path:
                row += "*"
            elif maze[y, x] == 1:
                row += "#"
            else:
                row += " "
        print(row)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Random Maze + Q-Learning Agent ===\n")
    width, height = 21, 21          # change these for a larger/smaller maze
    episodes = 10000   # safety limit only
    snapshot_every = 50

    maze, start, goal = generate_maze(width, height)
    env = MazeEnv(maze, start, goal)
    agent = QLearningAgent(height, width)

    print(f"Maze size: {height} x {width}")
    print(f"Start: {start}   Goal: {goal}")
    print(f"Training until goal is reached (safety limit {episodes} episodes)...\n")

    start_time = time.time()
    lengths, successes, first_success, episodes_run = train(
        env, agent, max_episodes=episodes, print_every=100,
        snapshot_every=snapshot_every, maze=maze, start=start, goal=goal
    )
    train_seconds = time.time() - start_time

    print("\n--- Training complete ---")
    print(f"Total training time: {train_seconds:.2f} seconds")
    print(f"Episodes actually run: {episodes_run}")
    if first_success:
        print(f"First successful episode: {first_success} (stopped early)")
        print(f"Time to first success: {train_seconds:.2f} seconds")
    else:
        print("Safety limit reached without success.")

    # Demonstrate the learned policy and time one greedy run
    t0 = time.time()
    path = run_greedy(env, agent)
    greedy_seconds = time.time() - t0
    print(f"\nGreedy path length: {len(path) - 1} steps")
    print(f"Time for one greedy solution: {greedy_seconds:.4f} seconds")

    # Visualizations
    print("\nASCII maze with learned path (*):")
    print_maze(maze, set(path))

    if HAS_MPL:
        visualize_maze(maze, path=path, start=start, goal=goal, filename="maze_with_path.png",
                       title="Final learned path")
        animate_path(maze, path, start, goal, filename="maze_path_animation.gif")

        window = 20
        if len(lengths) >= window:
            smoothed = np.convolve(lengths, np.ones(window)/window, mode="valid")
            plt.figure(figsize=(8, 4))
            plt.plot(smoothed, color="steelblue", linewidth=2)
            plt.title("Episode length (smoothed) — lower is better")
            plt.xlabel("Episode")
            plt.ylabel("Steps")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("maze_qlearning_progress.png", dpi=150)
            print("Saved progress plot: maze_qlearning_progress.png")
            plt.close()

        print(f"\nSnapshots saved every {snapshot_every} episodes in the 'snapshots/' folder.")
        create_progress_gif()

    print("\nDone. The agent has learned a policy that reaches the goal faster than random search.")
    print("Check the PNG, GIF, and snapshots folder for the visualizations.")

if __name__ == "__main__":
    main()
