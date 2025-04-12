import math
import time

import krpc
from krpc.client import Client
from krpc.services.spacecenter import Flight, SpaceCenter, VesselSituation
import numpy as np
from numpy.typing import NDArray

from flight_procedures.processing.machine_learning import Agent
from flight_procedures.processing.optimizer import calculate_reward
from flight_procedures.utils.classes import Coordinate
from flight_procedures.processing.data import observe_environment

FLIGHT_POINT_PREFIX: str = "FP-"
DEFAULT_ASL_ALTITUDE: float = 1000  # Meters above sea level
DEFAULT_CELESTIAL_BODY: str = "Kerbin"
DEFAULT_ADDRESS: str = "172.31.160.1"
ACTION_IDS: dict[int, str] = {
    0: "Throttle Up",
    1: "Throttle Down",
    2: "Roll Left",
    3: "Roll Right",
    4: "Yaw Left",
    5: "Yaw Right",
    6: "Pitch Up",
    7: "Pitch Down",
    8: "Do Nothing",
}
EPOCHS: int = 1000
DEFAULT_MARGINAL: float = 0.1

# Hyperparameters
GAMMA: float = 0.99
EPSILON: float = 1.0
LEARNING_RATE: float = 0.001
INPUT_SHAPE: int = 18
OUTPUT_SHAPE: int = 8
BATCH_SIZE: int = 32

# Normalization Constants
MAX_ASL_ALTITUDE: float = 70000  # Karman line in the simulation
MIN_ASL_ALTITUDE: float = 0  # Sea level
MACH_SPEED: float = 340.3  # Mach speed in meters per second

# Cost Function Constants
POSITIVE_REWARD: float = 1.0
NEGATIVE_REWARD: float = -1.0
CRASH_REWARD: float = NEGATIVE_REWARD * 100.0
MINIMUM_TWR: float = 0.5
MAXIMUM_TWR: float = 1.25
MINIMUM_ALTITUDE: float = 750  # Meters above sea level
MAXIMUM_ALTITUDE: float = 7500  # Meters above sea level
MAXIMUM_GFORCE: float = 2.0
MAXIMUM_AOA: float = 5.0
MAXIMUM_ROLL_ANGLE: float = 50.0  # Degrees
MAXIMUM_STALL_RATIO: float = 0.2  # Weight can't exceed 20% of Lift
MAXIMUM_PITCH_ANGLE: float = 20.0

# Weights
W_SPEED: float = 0.25
W_ALTITUDE: float = 3.0
W_POSITION: float = 1.0
W_GFORCE: float = 0.25
W_AOA: float = 0.25
W_ROLL_ANGLE: float = 0.45
W_STALL: float = 1.0
W_PITCH_ANGLE: float = 2.0


def get_client() -> Client:
    """
    Establishes a connection to the kRPC server and returns a Client object.

    Returns:
        Client: A kRPC Client object connected to the "Flight Procedure" server.
    """

    return krpc.connect(name="Flight Procedure", address=DEFAULT_ADDRESS)


def take_action(client: Client, action_id: int) -> None:
    """
    Executes a control action on the active vessel based on the given action ID.

    Args:
        client (Client): The kRPC client used to interact with the game.
        action_id (int): An integer representing the action to be taken. The actions are:
            0: Throttle Up
            1: Throttle Down
            2: Pitch Up
            3: Pitch Down
            4: Yaw Left
            5: Yaw Right
            6: Roll Left
            7: Roll Right
    """

    active_vessel = client.space_center.active_vessel

    match action_id:
        # Throttle up
        case 0:
            active_vessel.control.throttle += DEFAULT_MARGINAL

        # Throttle down
        case 1:
            active_vessel.control.throttle -= DEFAULT_MARGINAL

        # Pitch up
        case 2:
            active_vessel.control.pitch += DEFAULT_MARGINAL

        # Pitch down
        case 3:
            active_vessel.control.pitch -= DEFAULT_MARGINAL

        # Yaw left
        case 4:
            active_vessel.control.yaw -= DEFAULT_MARGINAL

        # Yaw right
        case 5:
            active_vessel.control.yaw += DEFAULT_MARGINAL

        # Roll left
        case 6:
            active_vessel.control.roll -= DEFAULT_MARGINAL

        # Roll right
        case 7:
            active_vessel.control.roll += DEFAULT_MARGINAL

        # do nothing
        case _:
            pass

    # print(f"Action Taken: {ACTION_IDS[action_id]}")


def has_crashed(client: Client) -> bool:
    """
    Checks if the active vessel has crashed.

    This function determines whether the active vessel's situation is
    different from 'flying', indicating a crash or landing.

    Args:
        client (Client): The kRPC client used to interact with the game.

    Returns:
        bool: True if the vessel has crashed or landed, False if it is still flying.
    """

    active_vessel = client.space_center.active_vessel
    if active_vessel.situation != VesselSituation.flying:
        return True
    return False


def interact_with_environment(
    action_id: int, destination: Coordinate, client: Client
) -> tuple[NDArray[np.float32], float, bool]:
    take_action(client=client, action_id=action_id)

    next_observation: NDArray[np.float32] = observe_environment(
        client=client, destination=destination
    )
    reward: float = calculate_reward(client=client, destination=destination)

    is_done: bool = False
    if has_crashed(client=client):
        is_done: bool = True
        reward: float = CRASH_REWARD

    return next_observation, reward, is_done


def check_if_arrived(client: Client) -> bool:
    return False


def start_over(client: Client) -> None:
    print("Starting over...")
    client.space_center.quickload()
    for second in range(5):
        print(f"Starting over in {5 - second} seconds...")
        time.sleep(1)
    main()


def load_agent() -> Agent:
    agent = Agent(
        gamma=GAMMA,
        epsilon=EPSILON,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
    )
    agent.load()
    return agent


def main():
    agent: Agent = load_agent()
    client: Client = get_client()
    destination: Coordinate = Coordinate(latitude=40.7128, longitude=-74.0060)

    is_done: bool = False
    iteration: int = 0
    reward_buffer: list[float] = []
    observation: NDArray[np.float32] = observe_environment(
        client=client, destination=destination
    )

    while not is_done:
        # network chooses an action
        action_id: int = agent.choose_action(observation=observation)

        # get environment response
        next_observation, reward, is_done = interact_with_environment(
            action_id=action_id,
            destination=destination,
            client=client,
        )
        reward_buffer.append(reward)

        # learn
        agent.store_transition(
            state=observation,
            action=action_id,
            reward=reward,
            next_state=next_observation,
            terminal=is_done,
        )
        agent.learn()

        # proceed to next episode
        observation = next_observation

        iteration += 1
        if iteration % 100 == 0:
            print(f"Iteration: {iteration}")
            print(f"Average reward: {sum(reward_buffer) / len(reward_buffer)}")

    agent.save()
    start_over(client=client)
