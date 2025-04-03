import random
import numpy as np

# Actions: 0 -> Rock, 1 -> Paper, 2 -> Scissors
actions = ["Rock", "Paper", "Scissors"]

# Q-table initialization (state-action values)
q_table = np.zeros((3, 3))  # State: Opponent's last move, Action: Agent's move

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate

def get_reward(agent_action, opponent_action):
    """Returns reward based on game result."""
    if agent_action == opponent_action:
        return 0  # Draw
    elif (agent_action - opponent_action) % 3 == 1:
        return 1  # Win
    else:
        return -1  # Loss

def choose_action(state):
    """Epsilon-greedy action selection."""
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 2)  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

def train(episodes=1000):
    """Train the agent using Q-learning."""
    last_opponent_action = random.randint(0, 2)  # Random initial opponent move
    
    for _ in range(episodes):
        agent_action = choose_action(last_opponent_action)
        opponent_action = random.randint(0, 2)  # Simulating a random opponent
        
        reward = get_reward(agent_action, opponent_action)
        
        # Q-learning update
        best_next_action = np.max(q_table[opponent_action])
        q_table[last_opponent_action, agent_action] += alpha * (reward + gamma * best_next_action - q_table[last_opponent_action, agent_action])
        
        last_opponent_action = opponent_action  # Update state for next round

def play():
    """Let the trained agent play against a human."""
    last_opponent_action = random.randint(0, 2)
    
    while True:
        print("\nChoose: 0 -> Rock, 1 -> Paper, 2 -> Scissors (or -1 to quit)")
        user_input = int(input("Your move: "))
        
        if user_input == -1:
            break
        elif user_input not in [0, 1, 2]:
            print("Invalid choice, try again.")
            continue
        
        agent_action = choose_action(last_opponent_action)
        print(f"Agent chose: {actions[agent_action]}")
        
        reward = get_reward(agent_action, user_input)
        if reward == 1:
            print("Agent wins!")
        elif reward == -1:
            print("You win!")
        else:
            print("It's a draw!")
        
        last_opponent_action = user_input  # Update state

# Train the agent
train(10000)

# Play against the trained agent
play()

#################################################################################################################3
# import random
# import numpy as np

# # Define actions
# actions = ['rock', 'paper', 'scissors']

# # Q-learning parameters
# alpha = 0.1  # Learning rate
# gamma = 0.9  # Discount factor
# epsilon = 0.2  # Exploration rate

# # Initialize Q-table
# Q = np.zeros((3, 3))  # State-action table

# def get_computer_action(prev_human_choice):
#     if random.uniform(0, 1) < epsilon:
#         return random.choice(actions)  # Explore
#     else:
#         return actions[np.argmax(Q[actions.index(prev_human_choice)])]  # Exploit

# def get_reward(human, computer):
#     if human == computer:
#         return 0  # Draw
#     elif (human == 'rock' and computer == 'scissors') or \
#          (human == 'scissors' and computer == 'paper') or \
#          (human == 'paper' and computer == 'rock'):
#         return -1  # Human wins, negative reward for AI
#     else:
#         return 1  # AI wins

# def update_q_table(prev_human_choice, human, computer):
#     reward = get_reward(human, computer)
#     state = actions.index(prev_human_choice)
#     action = actions.index(computer)
#     best_future_value = np.max(Q[action])
#     Q[state, action] = Q[state, action] + alpha * (reward + gamma * best_future_value - Q[state, action])

# def play_rps():
#     print("Welcome to Rock-Paper-Scissors with AI!")
#     prev_human_choice = random.choice(actions)  # Start with a random choice
#     while True:
#         human = input("Enter rock, paper, or scissors (or 'quit' to stop): ").lower()
#         if human == 'quit':
#             break
#         if human not in actions:
#             print("Invalid choice, try again.")
#             continue
        
#         computer = get_computer_action(prev_human_choice)
#         print(f"Computer chose: {computer}")
        
#         if human == computer:
#             print("It's a draw!")
#         elif get_reward(human, computer) == -1:
#             print("You win!")
#         else:
#             print("Computer wins!")
        
#         update_q_table(prev_human_choice, human, computer)
#         prev_human_choice = human  # Update state

# if _name_ == "_main_":
#     play_rps()