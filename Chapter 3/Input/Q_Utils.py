"""Utility functions used by Example 3.22 (Q-learning routing)."""

import matplotlib.pyplot as plt
import numpy as np


def showgraph(points_list):
    """Draw the routing graph described by a list of connected states."""
    nodes = sorted({node for route in points_list for node in route})
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    positions = {node: (np.cos(angle), np.sin(angle)) for node, angle in zip(nodes, angles)}

    plt.figure("Routing graph")
    for start, end in points_list:
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        plt.plot([x1, x2], [y1, y2], "k-", zorder=1)
    for node, (x, y) in positions.items():
        plt.scatter(x, y, s=500, color="skyblue", edgecolor="black", zorder=2)
        plt.text(x, y, str(node), ha="center", va="center", zorder=3)
    plt.title("Q-learning Routing Graph")
    plt.axis("equal")
    plt.axis("off")


def createRmat(matrix_size, points_list, goal):
    """Create a bidirectional reward matrix from the routing graph."""
    R = np.full((matrix_size, matrix_size), -1.0)
    for start, end in points_list:
        R[start, end] = 100 if end == goal else 0
        R[end, start] = 100 if start == goal else 0
    R[goal, goal] = 100
    return R


def available_actions(R, current_state):
    """Return state indices that can be reached from current_state."""
    return np.where(R[current_state] >= 0)[0]


def sample_next_action(available_act):
    """Choose a random valid next action."""
    return int(np.random.choice(available_act))


def update(R, Q, current_state, action, gamma):
    """Apply one Q-learning update and return the current highest Q value."""
    max_future_q = np.max(Q[action])
    Q[current_state, action] = R[current_state, action] + gamma * max_future_q
    return float(np.max(Q))
