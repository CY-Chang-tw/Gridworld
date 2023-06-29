from flask import Flask, render_template, request, jsonify
import numpy as np
import random

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/grid', methods=['POST'])
def grid():
    n = int(request.form['n'])
    gridworld = [['' for _ in range(n)] for _ in range(n)]
    return render_template('grid.html', gridworld=gridworld)

@app.route('/solve', methods=['POST'])
def solve():
    data = request.get_json()
    n = data['n']
    start = tuple(data['start'])
    end = tuple(data['end'])
    blocks = [tuple(block) for block in data['blocks']]

    # Run Q-learning algorithm
    solution_path = q_learning(n, start, end, blocks)

    return jsonify(solution_path)

def q_learning(n, start, end, blocks, episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    # Initialize Q-table
    q_table = np.zeros((n, n, 4))

    # Define actions and their corresponding coordinates
    actions = ['up', 'right', 'down', 'left']
    d_row = [-1, 0, 1, 0]
    d_col = [0, 1, 0, -1]

    # Q-learning algorithm
    for _ in range(episodes):
        row, col = start
        while (row, col) != end:
            action_idx = np.argmax(q_table[row, col]) if random.random() > epsilon else random.randint(0, 3)
            next_row, next_col = row + d_row[action_idx], col + d_col[action_idx]

            if 0 <= next_row < n and 0 <= next_col < n and (next_row, next_col) not in blocks:
                reward = 100 if (next_row, next_col) == end else -1
                q_table[row, col, action_idx] = q_table[row, col, action_idx] + alpha * (
                    reward + gamma * np.max(q_table[next_row, next_col]) - q_table[row, col, action_idx]
                )
                row, col = next_row, next_col

    # Extract solution path
    solution_path = []
    row, col = start
    while (row, col) != end:
        action_idx = np.argmax(q_table[row, col])
        next_row, next_col = row + d_row[action_idx], col + d_col[action_idx]
        if 0 <= next_row < n and 0 <= next_col < n and (next_row, next_col) not in blocks:
            solution_path.append([next_row, next_col])
            row, col = next_row, next_col

    return solution_path

if __name__ == '__main__':
    app.run(debug=True)
