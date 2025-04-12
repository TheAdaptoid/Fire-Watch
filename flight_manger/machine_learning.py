"""
This module contains an implementation of the Deep Q-Network agent.
"""

import os

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn, optim
from torch.nn import functional

DEFAULT_DIMENSIONALITY: int = 8
MEMORY_SIZE: int = 100_000
EPSILON_END: float = 0.001
EPSILON_DECAY: float = 5e-4

MODEL_IO_PATH: str = "model_weights/flight_plan_agent.pt"


class ModelLoadError(Exception):
    """Custom exception for model loading errors"""

    def __init__(self, *args):
        super().__init__(*args)


class DeepQNetwork(nn.Module):
    """Deep Q-Network with 3 hidden layers"""

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
        self.activation_function = functional.tanh
        self.normalizer = nn.BatchNorm1d(self.input_shape)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the neural network.

        Args:
            state (torch.Tensor): The input tensor representing the current state
                of the environment.

        Returns:
            torch.Tensor: The output tensor representing the action values for
                each possible action in the environment.
        """
        state = self.activation_function(self.layer1(state))
        state = self.activation_function(self.layer2(state))

        action = self.layer3(state)
        return action


class Agent:
    """Deep Q-Learning agent"""

    def __init__(
        self,
        gamma: float,
        epsilon: float,
        learning_rate: float,
        batch_size: int,
        input_shape: int,
        output_shape: int,
        memory_size: int = MEMORY_SIZE,
        eps_end: float = EPSILON_END,
        eps_decay: float = EPSILON_DECAY,
    ) -> None:
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
            self.memory_size, self.input_shape, dtype=torch.float32
        )
        self.new_state_memory = torch.zeros(
            self.memory_size, self.input_shape, dtype=torch.float32
        )
        self.action_memory = torch.zeros(self.memory_size, dtype=torch.int32)
        self.reward_memory = torch.zeros(self.memory_size, dtype=torch.float32)
        self.terminal_memory = torch.zeros(self.memory_size, dtype=torch.bool)

    def store_transition(
        self,
        state: NDArray[np.float32],
        action: int,
        reward: float,
        next_state: NDArray[np.float32],
        terminal: bool,
    ) -> None:
        """
        Stores the transition from the current state to the next state given the
        action and reward.

        Args:
            state: The current state of the environment.
            action: The action taken in the current state.
            reward: The reward received after taking the action in the current state.
            next_state: The next state of the environment after taking the action.
            terminal: Whether the episode terminated in the next state.
        """
        index: int = self.memory_counter % self.memory_size

        self.state_memeory[index] = torch.tensor(state, dtype=torch.float32)
        self.new_state_memory[index] = torch.tensor(next_state, dtype=torch.float32)
        self.action_memory[index] = torch.tensor(action, dtype=torch.int32)
        self.reward_memory[index] = torch.tensor(reward, dtype=torch.float32)
        self.terminal_memory[index] = torch.tensor(terminal, dtype=torch.bool)

        self.memory_counter += 1

    def choose_action(self, observation: NDArray[np.float32]) -> int:
        """
        Chooses an action based on the given observation of the environment.

        If the uniform random variable is greater than epsilon, the action is
        chosen by taking the argmax of the action values predicted by the
        neural network. Otherwise, the action is chosen randomly from the
        action space.

        Args:
            observation (NDArray[np.float32]): A numpy array representing the
                current state of the environment.

        Returns:
            int: The chosen action.
        """
        if np.random.uniform(0, 1) > self.epsilon:
            state = torch.tensor(observation).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return int(action)

    def learn(self) -> None:
        """
        Learns from the experience stored in the memory buffer.

        This method samples a random batch of transitions from the memory buffer,
        calculates the Q-targets using the Q-evaluation network, and updates the
        Q-evaluation network using the Q-learning loss function.

        The epsilon value is also updated after each learning iteration.
        """
        if self.memory_counter < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        max_memory: int = min(self.memory_size, self.memory_counter)
        batch = np.random.choice(max_memory, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = self.state_memeory[batch].clone().detach().to(self.q_eval.device)
        new_state_batch = (
            self.new_state_memory[batch].clone().detach().to(self.q_eval.device)
        )
        reward_batch = self.reward_memory[batch].clone().detach().to(self.q_eval.device)
        terminal_batch = (
            self.terminal_memory[batch].clone().detach().to(self.q_eval.device)
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

    def save(self, path: str = MODEL_IO_PATH) -> None:
        """
        Saves the current model state to the specified path.

        This method serializes the state dictionary of the neural network and writes
        it to the provided file path using PyTorch's save functionality.

        Args:
            path (str, optional): The file path where the model should be saved.
                Defaults to MODEL_IO_PATH.
        """

        torch.save(
            obj=self.q_eval.state_dict(),
            f=path,
        )

    def load(self, path: str = MODEL_IO_PATH) -> None:
        """
        Loads a saved model from the given path.

        If the path does not exist, no action is taken.

        Args:
            path (str): The path to the saved model. Defaults to MODEL_IO_PATH.

        Raises:
            ModelLoadError: If loading the model fails.
        """
        if not os.path.exists(path):
            return

        try:
            self.q_eval.load_state_dict(
                state_dict=torch.load(f=path, weights_only=True)
            )
        except ModelLoadError as e:
            print(f"Failed to load model: {e}")
