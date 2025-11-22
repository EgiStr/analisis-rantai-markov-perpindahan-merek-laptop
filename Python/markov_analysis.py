import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eig
import os

# Load data
data_path = os.path.join(os.path.dirname(__file__), '../data/datapemstok.csv')
data = pd.read_csv(data_path, encoding='latin1')

# Select relevant columns
data_clean = data.iloc[:, [4, 5, 6]]  # Columns: first_brand, still_same, current_brand
data_clean.columns = ['first_brand', 'still_same', 'current_brand']

# Handle NA and logic
data_clean = data_clean.dropna(subset=['first_brand'])
data_clean['current_brand'] = data_clean.apply(
    lambda row: row['first_brand'] if row['still_same'] == 'Ya' else row['current_brand'], axis=1
)
data_clean = data_clean.dropna(subset=['current_brand'])

print("Cleaned data:")
print(data_clean.head())

# States
states = sorted(set(data_clean['first_brand'].unique()) | set(data_clean['current_brand'].unique()))
state_index = {state: i for i, state in enumerate(states)}
print("States:", states)

# Transition matrix
n = len(states)
transition_matrix = np.zeros((n, n))

for _, row in data_clean.iterrows():
    i = state_index[row['first_brand']]
    j = state_index[row['current_brand']]
    transition_matrix[i, j] += 1

# Normalize rows
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
transition_matrix = np.nan_to_num(transition_matrix)  # Handle zero rows

print("Transition Matrix:")
print(pd.DataFrame(transition_matrix, index=states, columns=states))

# Visualize transition diagram
G = nx.DiGraph()
for i in range(n):
    for j in range(n):
        if transition_matrix[i, j] > 0:
            G.add_edge(states[i], states[j], weight=transition_matrix[i, j])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
edge_labels = {(states[i], states[j]): f"{transition_matrix[i, j]:.2f}" for i in range(n) for j in range(n) if transition_matrix[i, j] > 0}
nx.draw_networkx_edge_labels(G, pos, edge_labels)
plt.title("Markov Chain Transition Diagram")
output_path = os.path.join(os.path.dirname(__file__), '../output/transition_diagram.png')
plt.savefig(output_path)
# plt.show()  # Remove for non-interactive

# n-step probabilities
n_step = 3
P_n = np.linalg.matrix_power(transition_matrix, n_step)
print(f"Probabilities after {n_step} steps:")
print(pd.DataFrame(P_n, index=states, columns=states))

# Steady-state distribution
# Solve (P - I) pi = 0 with sum pi = 1
P_T = transition_matrix.T
evals, evecs = eig(P_T)
# Find eigenvector for eigenvalue 1
idx = np.argmin(np.abs(evals - 1))
steady_state = np.real(evecs[:, idx])
steady_state = steady_state / steady_state.sum()
print("Steady-state distribution:")
for state, prob in zip(states, steady_state):
    print(f"{state}: {prob:.4f}")

# Plot steady state
plt.bar(states, steady_state)
plt.title("Steady-State Distribution")
plt.ylabel("Probability")
plt.xticks(rotation=45)
output_path2 = os.path.join(os.path.dirname(__file__), '../output/steady_state.png')
plt.savefig(output_path2)
# plt.show()

# Classification: Simple check for absorbing states
absorbing = []
for i, state in enumerate(states):
    if transition_matrix[i, i] == 1 and np.sum(transition_matrix[i, :]) == 1:
        absorbing.append(state)
print("Absorbing states:", absorbing)

# Simulation
def simulate_markov_chain(start_state, steps):
    current = state_index[start_state]
    path = [states[current]]
    for _ in range(steps):
        current = np.random.choice(n, p=transition_matrix[current])
        path.append(states[current])
    return path

sim_path = simulate_markov_chain("Lenovo", 10)
print("Simulation path starting from Lenovo:", sim_path)