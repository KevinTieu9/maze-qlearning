# Maze Q-Learning

**Author:** Kevin Tieu

## Overview

Generates a random maze by recursive backtracking and trains a tabular Q-learning agent. Path length decreases over successive episodes as the agent learns a policy from start to goal.

## What the Program Does

The script builds a maze, places a start and a goal, and runs episodes of Q-learning. The agent receives a reward for reaching the goal and a small penalty per step. A progress plot shows episode length over training.

## Files

| File | Description |
|------|-------------|
| `maze_qlearning.py` | Maze generation and Q-learning |

## Techniques

- Maze generation (recursive backtracking)
- Tabular Q-learning
- Reinforcement learning

## Run

```bash
python3 maze_qlearning.py
```
