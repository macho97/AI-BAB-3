# Exercise 3.16: Q-learning for an eight-state routing diagram
import numpy as np

# Routes: 0 -> 1 -> (2 or 3) -> (4 or 5) -> 6 -> 7 (goal)
routes = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6), (6, 7)]
size, goal, gamma = 8, 7, 0.8
R = np.full((size, size), -1.0)
for start, end in routes:
    R[start, end] = 100 if end == goal else 0
    R[end, start] = 0
R[goal, goal] = 100
Q = np.zeros((size, size))

for _ in range(1000):
    state = np.random.randint(size)
    while state != goal:
        actions = np.where(R[state] >= 0)[0]
        action = np.random.choice(actions)
        Q[state, action] = R[state, action] + gamma * Q[action].max()
        state = action

state, path = 0, [0]
while state != goal:
    state = np.argmax(Q[state])
    path.append(int(state))
print("Trained Q matrix:\n", Q / Q.max() * 100)
print("Most efficient path:", path)
