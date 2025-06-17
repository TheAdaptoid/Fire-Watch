"""
This module provides functions for interacting with the environment and training
a neural network model using the Q-learning algorithm.
"""

import time
from random import randint

from flight_manger.cost_functions import calculate_reward
from flight_manger.data import Observation, observe
from flight_manger.krpc_client import (
    Client,
    create_waypoint,
    delete_waypoints,
    manipulate_controls,
)
from flight_manger.machine_learning import Agent
from flight_manger.utils import Coordinate, create_logger

# Hyperparameters
GAMMA: float = 0.99
EPSILON: float = 1.0
LEARNING_RATE: float = 0.001
OUTPUT_SHAPE: int = 9
BATCH_SIZE: int = 128


def interact(
    client: Client, action_id: int, target: Coordinate
) -> tuple[Observation, float, bool]:
    """
    Interacts with the environment to gather new observations and rewards.

    This function adjusts the flight controls of the vessel according to the
    specified action, observes the new state of the environment, calculates the
    reward for the action, and determines if the episode is complete.

    Args:
        client (Client): The kRPC client used to interface with the game.
        action_id (int): The ID of the action to take.
        target (Coordinate): The target coordinate for flight navigation.

    Returns:
        tuple[Observation, float, bool]: A tuple containing the new observation,
            the reward for the action, and a boolean indicating if the episode is
            complete.
    """
    # Adjust flight controls
    manipulate_controls(client=client, action_id=action_id)

    # Get new observation
    observation: Observation = observe(client=client, target=target)

    # Calculate reward and check if episode is done
    reward, is_done = calculate_reward(client=client, target=target)

    return observation, reward, is_done


def load_ann(input_shape: int) -> Agent:
    """
    Initializes and loads a neural network agent for flight control.

    This function creates an `Agent` instance with specified hyperparameters,
    including the input shape, and loads its pre-trained weights from storage.

    Args:
        input_shape (int): The number of input features for the neural network.

    Returns:
        Agent: An instance of the `Agent` class with loaded weights.
    """

    ann: Agent = Agent(
        gamma=GAMMA,
        epsilon=EPSILON,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        input_shape=input_shape,
        output_shape=OUTPUT_SHAPE,
    )
    ann.load()
    return ann


def start_training(client: Client) -> None:
    """
    Prepares the simulation and starts the training process.

    This function is a wrapper for the `train` function and is intended to be used
    as an entry point for the training process. It reloads the most recent
    simulation and waits 5 seconds before starting the training process.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.
    """
    client.space_center.quickload()
    for second in range(5):
        print(f"Starting training in {5 - second} seconds...")
        time.sleep(1)
    train(client=client)


def train(client: Client) -> None:
    """
    Trains a neural network agent to control a simulated flight.

    This function sets up a target position for the flight, initializes the
    neural network agent, and manages state and reward tracking. It iteratively
    chooses actions based on observations, interacts with the environment to
    gather new observations and rewards, stores transitions for learning, and
    saves the model periodically. The training continues until the episode is
    complete, after which it saves the final model and restarts the training
    process.

    Args:
        client (Client): The kRPC client used to interface with the simulation.

    """

    # Set target position
    target_position: Coordinate = Coordinate(latitude=randint(-15, 20), longitude=-40)
    delete_waypoints(client=client)
    create_waypoint(client=client, coordinate=target_position)
    print(f"Target position: {target_position}")

    # Get initial observation
    observation: Observation = observe(client=client, target=target_position)

    # Create neural network
    input_shape: int = observation.length
    ann: Agent = load_ann(input_shape=input_shape)

    # state management
    logger = create_logger()
    is_done: bool = False
    iteration: int = 0
    reward_buffer: list[float] = []

    while not is_done:
        # network chooses an action
        action_id: int = ann.choose_action(
            observation=observation.to_ndarray(normalize=True)
        )

        # interact with environment
        next_observation, reward, is_done = interact(
            client=client, action_id=action_id, target=target_position
        )

        # store the transition in memory and learn
        ann.store_transition(
            state=observation.to_ndarray(normalize=True),
            action=action_id,
            reward=reward,
            next_state=next_observation.to_ndarray(normalize=True),
            terminal=is_done,
        )
        ann.learn()

        # update observation
        observation = next_observation

        # update iteration and reward buffer
        iteration += 1
        reward_buffer.append(reward)
        if iteration % 100 == 0:
            buffer_sum: float = sum(reward_buffer)
            buffer_size: int = len(reward_buffer)
            print(
                f"iteration: {iteration}, "
                + f"epsilon: {round(ann.epsilon, 3)}, "
                + f"mean reward: {buffer_sum / buffer_size:.3f}"
            )
            reward_buffer.clear()

            # quick save
            ann.save()

        # check if episode is done
        if is_done:
            logger.warning(
                "Episode finished after %s iterations. Last reward: %s",
                iteration,
                reward,
            )
            break

    # save model
    ann.save()

    # start over
    start_training(client=client)
