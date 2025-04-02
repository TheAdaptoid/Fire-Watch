import numpy as np
from numpy.typing import NDArray

import torch
from torch import nn
from torch.nn import functional
from torch import optim

INPUT_SHAPE: int = 18
OUTPUT_SHAPE: int = 8
DEFAULT_DIMENSIONALITY: int = 128
MEMORY_SIZE: int = 100_000
EPSILON_END: float = 0.01
EPSILON_DECAY: float = 5e-4


class DeepQNetwork(nn.Module):
    def __init__(self, learning_rate: float, input_shape: int, output_shape: int):
        super(DeepQNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.layer1 = nn.Linear(self.input_shape, DEFAULT_DIMENSIONALITY)
        self.layer2 = nn.Linear(DEFAULT_DIMENSIONALITY, DEFAULT_DIMENSIONALITY)
        self.layer3 = nn.Linear(DEFAULT_DIMENSIONALITY, self.output_shape)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        state = functional.relu(self.layer1(state))
        state = functional.relu(self.layer2(state))
        action = self.layer3(state)
        return action


class Agent:
    def __init__(
        self,
        gamma: float,
        epsilon: float,
        learning_rate: float,
        input_shape: int,
        output_shape: int,
        batch_size: int,
        memory_size: int = MEMORY_SIZE,
        eps_end: float = EPSILON_END,
        eps_decay: float = EPSILON_DECAY,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.action_space = [i for i in range(self.output_shape)]
        self.memory_counter = 0

        # Network
        self.q_eval = DeepQNetwork(
            learning_rate=self.learning_rate,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
        )

        # Memory
        self.state_memeory = torch.zeros(
            self.memory_size, self.input_shape, dtype=np.float32
        )
        self.new_state_memory = torch.zeros(
            self.memory_size, self.input_shape, dtype=np.float32
        )
        self.action_memory = torch.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = torch.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = torch.zeros(self.memory_size, dtype=np.bool)

    def store_transition(
        self,
        state: NDArray[np.float32],
        action: int,
        reward: float,
        next_state: NDArray[np.float32],
        terminal: bool,
    ) -> None:
        index: int = self.memory_counter % self.memory_size

        self.state_memeory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal

        self.memory_counter += 1

    def choose_action(self, observation: NDArray[np.float32]) -> int:
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return int(action)

    def learn(self) -> None:
        if self.memory_counter > self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        max_memory: int = min(self.memory_size, self.memory_counter)
        batch = np.random.choice(max_memory, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memeory[batch]).to(self.q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(
            self.q_eval.device
        )
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(
            self.q_eval.device
        )

        action_batch = self.action_memory[batch]

        q_eval = self.q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0  # set q_next to 0 if terminal
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.q_eval.loss_function(q_target, q_eval).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.epsilon = (
            self.epsilon - self.eps_decay
            if self.epsilon > self.eps_end
            else self.eps_end
        )
