import math

import krpc
from krpc.client import Client
from krpc.services.spacecenter import Flight, SpaceCenter
import numpy as np
from numpy.typing import NDArray

from flight_procedures.processing.machine_learning import Agent
from flight_procedures.utils import (
    FlightDataPacket,
    FlightPlan,
    FlightPoint,
    timed_function,
)

FLIGHT_POINT_PREFIX: str = "FP-"
DEFAULT_ASL_ALTITUDE: float = 1000  # Meters above sea level
MINIMUM_AGL_ALTITUDE: float = 750  # Meters
MAXIMUM_AGL_ALTITUDE: float = 3500  # Meters
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
}
EPOCHS: int = 1000
POSITIVE_REWARD: float = 1.0
NEGATIVE_REWARD: float = -1.0

GAMMA: float = 0.99
EPSILON: float = 1.0
LEARNING_RATE: float = 0.001
INPUT_SHAPE: int = 18
OUTPUT_SHAPE: int = 8
BATCH_SIZE: int = 32


def get_client() -> Client:
    """
    Establishes a connection to the kRPC server and returns a Client object.

    Returns:
        Client: A kRPC Client object connected to the "Flight Procedure" server.
    """

    return krpc.connect(name="Flight Procedure", address=DEFAULT_ADDRESS)


def delete_flight_path_waypoints(client: Client) -> None:
    """
    Deletes all waypoints with names starting with the `FLIGHT_POINT_NAME_PREFIX`.

    Args:
        client (Client): The kRPC client.
    """

    for waypoint in client.space_center.waypoint_manager.waypoints:
        name: str = waypoint.name
        if name.startswith(FLIGHT_POINT_PREFIX):
            waypoint.remove()


def create_flight_path_waypoints(client: Client, flight_plan: FlightPlan) -> None:
    """
    Creates waypoints for each flight point in the provided flight plan.
    The waypoints are named using a prefix followed by the latitude and
    longitude of the flight point.
    Each waypoint is created at the `DEFAULT_ALTITUDE` and
    is associated with the `DEFAULT_CELESTIAL_BODY`.

    Args:
        client (Client): The kRPC client used to interact with the game.
        flight_plan (FlightPlan): The flight plan containing the route set of flight points.
    """

    for index, flight_point in enumerate(flight_plan.route_set):
        flight_point.altitude = DEFAULT_ASL_ALTITUDE
        client.space_center.waypoint_manager.add_waypoint(
            latitude=flight_point.latitude,
            longitude=flight_point.longitude,
            body=SpaceCenter(client=client).bodies[DEFAULT_CELESTIAL_BODY],
            name=f"{FLIGHT_POINT_PREFIX}{index}:{flight_point.latitude},{flight_point.longitude}",
        )


@timed_function
def generate_waypoints(client: Client, flight_plan: FlightPlan) -> None:
    """
    Generates waypoints for a given flight plan by first deleting any existing
    waypoints with names starting with the `FLIGHT_POINT_NAME_PREFIX`, and then
    creating new waypoints for each flight point in the flight plan.

    Args:
        client (Client): The kRPC client used to interact with the game.
        flight_plan (FlightPlan): The flight plan containing the route set of flight points.
    """

    delete_flight_path_waypoints(client=client)
    create_flight_path_waypoints(client=client, flight_plan=flight_plan)


def get_flight_data(client: Client) -> FlightDataPacket:
    """
    Retrieves the current flight data from the game and returns it as a
    FlightDataPacket object.

    The retrieved data includes the altitude above sea level, altitude above
    ground level, latitude, longitude, pitch, heading, roll, X, Y, and Z
    components of the velocity vector, the magnitude of the velocity vector,
    the horizontal and vertical speeds, the g-force, and the angle of attack.

    Args:
        client (Client): The kRPC client used to interact with the game.

    Returns:
        FlightDataPacket: The current flight data as a FlightDataPacket object.
    """
    flight_data: Flight = client.space_center.active_vessel.flight()
    velocity_vector: tuple[float, float, float] = flight_data.velocity

    return FlightDataPacket(
        altitude_asl=flight_data.mean_altitude,
        altitude_agl=flight_data.surface_altitude,
        latitude=flight_data.latitude,
        longitude=flight_data.longitude,
        pitch=flight_data.pitch,
        heading=flight_data.heading,
        roll=flight_data.roll,
        velocity_x=velocity_vector[0],
        velocity_y=velocity_vector[1],
        velocity_z=velocity_vector[2],
        velocity_magnitude=flight_data.speed,
        horizontal_speed=flight_data.horizontal_speed,
        vertical_speed=flight_data.vertical_speed,
        g_force=flight_data.g_force,
        aoa=flight_data.angle_of_attack,
    )


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
    current_throttle: float = active_vessel.control.throttle
    current_pitch: float = active_vessel.control.pitch
    current_yaw: float = active_vessel.control.yaw
    current_roll: float = active_vessel.control.roll

    match action_id:
        # Throttle up
        case 0:
            active_vessel.control.throttle = current_throttle + 0.1

        # Throttle down
        case 1:
            active_vessel.control.throttle = current_throttle - 0.1

        # Pitch up
        case 2:
            active_vessel.control.pitch = current_pitch + 0.1

        # Pitch down
        case 3:
            active_vessel.control.pitch = current_pitch - 0.1

        # Yaw left
        case 4:
            active_vessel.control.yaw = current_yaw + 0.1

        # Yaw right
        case 5:
            active_vessel.control.yaw = current_yaw - 0.1

        # Roll left
        case 6:
            active_vessel.control.roll = current_roll + 0.1

        # Roll right
        case 7:
            active_vessel.control.roll = current_roll - 0.1

    # print(f"Action Taken: {ACTION_IDS[action_id]}")


def calculate_distance(
    lat1: float, lon1: float, alt1: float, lat2: float, lon2: float, alt2: float
) -> float:
    """
    Calculates the Euclidean distance between two points specified by
    latitude, longitude, and altitude coordinates.

    Args:
        lat1 (float): The latitude of the first point.
        lon1 (float): The longitude of the first point.
        alt1 (float): The altitude of the first point.
        lat2 (float): The latitude of the second point.
        lon2 (float): The longitude of the second point.
        alt2 (float): The altitude of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """

    return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2 + (alt2 - alt1) ** 2)



def determine_reward(
    client: Client,
    target_point: FlightPoint,
    old_flight_data: NDArray[np.float32],
    new_flight_data: NDArray[np.float32],
) -> float:
    """
    Determines the reward based on the change in distance to the target point.

    Args:
        client (Client): The kRPC client used to interact with the game.
        target_point (FlightPoint): The target flight point the aircraft is trying to reach.
        old_flight_data (NDArray[np.float32]): The previous flight data,
            including position and altitude.
        new_flight_data (NDArray[np.float32]): The current flight data,
            including position and altitude.

    Returns:
        float: A reward value indicating improvement (or lack thereof) in reaching the target point.
            Returns a positive reward if the new position is closer to the target compared to
            the old position, otherwise returns a negative reward.
    """
    if check_if_too_low_or_high(client=client):
        return NEGATIVE_REWARD

    old_position = FlightPoint(
        latitude=old_flight_data[2],
        longitude=old_flight_data[3],
        altitude=old_flight_data[0],
    )
    new_position = FlightPoint(
        latitude=new_flight_data[2],
        longitude=new_flight_data[3],
        altitude=new_flight_data[0],
    )

    prev_distance = calculate_distance(
        old_position.latitude,
        old_position.longitude,
        old_position.altitude,
        target_point.latitude,
        target_point.longitude,
        target_point.altitude,
    )
    new_distance = calculate_distance(
        new_position.latitude,
        new_position.longitude,
        new_position.altitude,
        target_point.latitude,
        target_point.longitude,
        target_point.altitude,
    )

    if new_distance < prev_distance:
        return POSITIVE_REWARD
    else:
        return NEGATIVE_REWARD


def environment_response(
    client: Client,
    target_point: FlightPoint,
    old_flight_data: NDArray[np.float32],
    action_id: int,
) -> tuple[NDArray[np.float32], float, bool]:
    # Take action in environment
    """
    Takes an action in the environment and returns the next state, reward,
    and if the episode is over.

    Args:
        client (Client): The kRPC client used to interact with the game.
        target_point (FlightPoint): The target flight point the aircraft is trying to reach.
        old_flight_data (NDArray[np.float32]): The previous flight data,
            including position and altitude.
        action_id (int): The action id to take in the environment.

    Returns:
        tuple[NDArray[np.float32], float, bool]: A tuple containing the new flight data,
            a reward value, and a boolean indicating if the episode is over.
    """

    take_action(client=client, action_id=action_id)

    # Get response from environment
    new_flight_data = make_observation(client=client, target_point=target_point)
    reward = determine_reward(
        client=client,
        target_point=target_point,
        old_flight_data=old_flight_data,
        new_flight_data=new_flight_data,
    )

    # Check if episode is over
    is_terminal: bool = False
    if check_if_crashed(client=client) or check_if_arrived(client=client):
        is_terminal = True

    return new_flight_data, reward, is_terminal


def check_if_crashed(client: Client) -> bool:
    return False


def check_if_arrived(client: Client) -> bool:
    return False

def check_if_too_low_or_high(client: Client) -> bool:
    altitude = client.space_center.active_vessel.flight().surface_altitude
    if altitude > MAXIMUM_AGL_ALTITUDE or altitude < MINIMUM_AGL_ALTITUDE:
        return True
    return False

def make_observation(client: Client, target_point: FlightPoint) -> NDArray[np.float32]:
    """
    Creates an observation vector by combining the current flight
    data with the target point coordinates.

    Args:
        client (Client): The kRPC client used to access the current flight data.
        target_point (FlightPoint): The target point to which the aircraft is trying to fly.

    Returns:
        NDArray[np.float32]: An array containing the flight data vector followed by
            the latitude, longitude, and altitude of the target point.
    """

    flight_data = get_flight_data(client=client)
    flight_data_vector = flight_data.to_vector()
    flight_data_vector.append(target_point.latitude)
    flight_data_vector.append(target_point.longitude)
    flight_data_vector.append(target_point.altitude)

    return np.array(flight_data_vector, dtype=np.float32)


@timed_function
def execute_flight_plan(client: Client, flight_plan: FlightPlan) -> None:
    agent = Agent(
        gamma=GAMMA,
        epsilon=EPSILON,
        learning_rate=LEARNING_RATE,
        input_shape=INPUT_SHAPE,
        output_shape=OUTPUT_SHAPE,
        batch_size=BATCH_SIZE,
    )
    agent.load()

    for fp_index, flight_point in enumerate(flight_plan.route_set):
        print(f"Executing flight point {fp_index}")
        print(f"Target point: {flight_point}")

        # TODO: Reset the environment
        is_done: bool = False
        reward_buffer: list[float] = []
        observation = make_observation(client=client, target_point=flight_point)

        # Execute the flight plan
        while not is_done:
            # Take action
            action_id: int = agent.choose_action(observation=observation)

            # Get environment response
            new_observation, reward, done = environment_response(
                client=client,
                action_id=action_id,
                old_flight_data=observation,
                target_point=flight_point,
            )

            # Store the transition in memory
            agent.store_transition(
                state=observation,
                action=action_id,
                reward=reward,
                next_state=new_observation,
                terminal=done,
            )

            # Learn
            agent.learn()

            # Update the observation
            observation = new_observation

            # Check training progress
            reward_buffer.append(reward)
            if len(reward_buffer) % 100 == 0:
                print(f"Average reward: {sum(reward_buffer) / len(reward_buffer)}")
                reward_buffer = []
                agent.save()

        # Print progress
        if 100 % fp_index == 0:
            print(
                f"Completed {fp_index} flight points out of {len(flight_plan.route_set)}"
            )
