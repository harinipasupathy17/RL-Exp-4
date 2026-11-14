# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement


Implement the Value Iteration algorithm to find the optimal value function V∗(s) and optimal policy π∗(s)(s) for the FrozenLake environment modeled as a finite Markov Decision Process.


## Software Requirements
1. Python 3.x
2. Gymnasium library
3. NumPy
4. Matplotlib
5. Jupyter Notebook / Google Colab


## Environment Description

The FrozenLake environment is a grid-world problem where:
S → Starting state
F → Frozen surface (safe state)
H → Hole (terminal state with reward 0)
G → Goal state (terminal state with reward 1)

The agent must move from the start state to the goal while avoiding holes.

The environment is a 4 × 4 grid:
```

F S F F
F H F H
F F F H
G H F F 
```

The environment is stochastic because is_slippery=True, meaning the agent may not always move in the intended direction.

## MDP Representation

The FrozenLake environment can be represented as:

MDP=(S,A,P,R,γ)
where:
S → Set of states (16 states)
A → Set of actions (Left, Down, Right, Up)
P → Transition probability function
R → Reward function
γ → Discount factor

Actions:
0 - Left
1 -	Down
2	- Right
3	- Up

## Theory


Value Iteration is a dynamic programming algorithm used to find the optimal policy in a Markov Decision Process. It repeatedly updates state values by choosing the best possible action until the values converge. The final values are used to generate the optimal policy for the agent.



## Algorithm

Step 1:
Initialize the value function: V(s)=0 for all states.

Step 2:
For every state, calculate the maximum expected value over all actions.

Step 3:
Update the value function using Bellman optimality equation.

Step 4:
Repeat until the change in values is less than the threshold θ.

Step 5:
Extract the optimal policy by selecting the action with maximum value.



## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env_desc = [
    "FSFF",
    "FHFH",
    "FFFH",
    "GFFF"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)

# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Performs Value Iteration and returns:
    Value Function,
    Optimal Policy,
    Number of Iterations
    """

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize Value Function
    V = np.zeros(n_states)
    iteration = 0
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    action_values[a] += prob * (
                        reward + gamma * V[next_state]
                    )
            V[s] = np.max(action_values)
            delta = max(delta, abs(v - V[s]))
        iteration += 1

        if delta < theta:
            break
    # Extract Optimal Policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                action_values[a] += prob * (
                    reward + gamma * V[next_state]
                )
        policy[s] = np.argmax(action_values)
    return V, policy, iteration

# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------
V, policy, iterations = value_iteration(env)


# -------------------------------------------------
# Display Output
# -------------------------------------------------
print("Name: Harini P ")
print("Register Number:  212224230082 ")
print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))


action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()



```

---

## Output

for this : the FrozenLake environment is
```
F S F F
F H F H
F F F H
G F F F
```

<img width="643" height="246" alt="image" src="https://github.com/user-attachments/assets/3485cc4e-8f90-45a6-bb75-683d0f6f00aa" />

for this : the FrozenLake environment is
```
S F F F
F H F H
F F F H
H F F G
```
<img width="562" height="297" alt="image" src="https://github.com/user-attachments/assets/079bc15a-e122-4eb9-956d-376d9b9da4cf" />




---

## Result
```text
## Result

The Value Iteration algorithm was implemented successfully using the Gymnasium FrozenLake-v1 environment. The optimal state-value function and optimal policy were obtained, enabling the agent to select the best actions to reach the goal while avoiding hazardous states.

```
---

## Inference
```text
The Value Iteration algorithm successfully computed the optimal state values and policies for both FrozenLake environments. The difference in the number of iterations and optimal policies is due to the changes in grid arrangement, hole positions, and goal location. The results show that Value Iteration can adapt to different environments and find an efficient path from the starting state to the goal while avoiding unsafe states.

```
---
