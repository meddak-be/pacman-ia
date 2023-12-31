# Pac-Man AI Project

## Overview

This project focuses on developing an intelligent Pac-Man agent using various AI techniques. The project is divided into **four** phases, each building upon the previous to enhance Pac-Man's decision-making abilities in the game environment. 

I did this project in the context of the **Artifical Intelligence** course given at ULB.

### Table of Contents

1. Part 1: Pathfinding Algorithms
2. Part 2: Adversarial Search
3. Part 3: Bayesian Inference
4. Part 4: Evaluating State and Action

## Part 1: Pathfinding Algorithms

### Description

In this phase, we focus on implementing classic search algorithms to enable Pac-Man to calculate the optimal path for collecting pellets and avoiding ghosts.

#### Features

- Depth-First Search (DFS)
- Breadth-First Search (BFS)
- A* Search Algorithm

#### How to Run


```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

## Part 2: Adversarial Search

### Description

This phase introduces adversarial search algorithms considering the presence of ghosts as adversaries.

#### Features

- Minimax Algorithm
- Alpha-Beta Pruning
- Expectimax Algorithm

#### How to Run


```bash
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3 python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
```

## Part 3: Bayesian Inference

### Description

In this part, we implement algorithms for Bayesian inference to predict the presence of ghosts under uncertainty.

#### Features

- Exact Inference using Bayes' Theorem
- Approximate Inference Algorithms

#### How to Run



```bash
python pacman.py -l smallGrid -p BayesAgent python pacman.py -l trickyClassic -p BayesAgent
```

## Part 4: Evaluating State and Action

### Description

The final phase involves the implementation of algorithms based on evaluating the quality of a state or action.

#### Features

- Reinforcement Learning
- Value Iteration
- Q-Learning

#### How to Run


```bash
python pacman.py -p QLearningAgent -x 2000 -n 2010 -l smallGrid python pacman.py -p ValueIterationAgent -a depth=3 -l mediumClassic
```

## Installation

- Python 3.x
- Pygame (for GUI)

## Usage

Run the commands specified in each part's section to see Pac-Man in action using different AI strategies.