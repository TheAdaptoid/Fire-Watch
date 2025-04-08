import math
import time
from traceback import print_tb

import krpc
from krpc.client import Client
from krpc.services.spacecenter import Flight, SpaceCenter, VesselSituation
import numpy as np
from numpy.typing import NDArray

from flight_procedures.processing.machine_learning import Agent
from flight_procedures.utils import (
    FlightDataPacket,
    FlightPlan,
    FlightPoint,
    timed_function
)

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


def normalize_altitude(altitude: float) -> float:
    """
    Normalizes an altitude value to a float between 0 and 1.

    Args:
        altitude (float): The altitude value to be normalized.

    Returns:
        float: The normalized altitude value.
    """
    return (altitude - MIN_ASL_ALTITUDE) / (MAX_ASL_ALTITUDE - MIN_ASL_ALTITUDE)


def normalize_speed(speed: float) -> float:
    """
    Normalizes a speed value as a float representing the percentage of Mach speed.
    i.e. 1.2 is 120% of the speed of sound, 0.5 is half the speed of sound, etc.

    Args:
        speed (float): The speed value to be normalized.

    Returns:
        float: The normalized speed value.
    """
    return float(speed / MACH_SPEED)


def normalize_angle(angle: float) -> float:
    """
    Normalizes an angle value as a float between 0 and 1.

    Args:
        angle (float): The angle value to be normalized.

    Returns:
        float: The normalized angle value.
    """
    return float(angle / 360)


def velocity_unit_vector(
    velocity_vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    Calculates the unit vector of the provided velocity vector.

    Args:
        velocity_vector (tuple[float, float, float]): The velocity vector as a tuple of 3 floats.

    Returns:
        tuple[float, float, float]: The unit vector of the provided
            velocity vector as a tuple of 3 floats.
    """
    x: float = velocity_vector[0]
    y: float = velocity_vector[1]
    z: float = velocity_vector[2]

    magnitude: float = math.sqrt(x**2 + y**2 + z**2)

    if magnitude == 0:
        return (0, 0, 0)

    return (x / magnitude, y / magnitude, z / magnitude)


def normalize_flight_data(flight_data: FlightDataPacket) -> FlightDataPacket:
    """
    Normalizes the various parameters of a FlightDataPacket object.

    This function adjusts the altitude, speed, angle, and velocity vector
    components of the FlightDataPacket to normalized values. Altitude values
    are scaled between 0 and 1, speeds are normalized as percentages of Mach
    speed, angles are normalized between 0 and 1, and the velocity components
    are converted to a unit vector.

    Args:
        flight_data (FlightDataPacket): The flight data to be normalized.

    Returns:
        FlightDataPacket: The normalized flight data.
    """
    # Altitudes
    flight_data.altitude_asl = normalize_altitude(flight_data.altitude_asl)
    flight_data.altitude_agl = normalize_altitude(flight_data.altitude_agl)

    # Speeds
    flight_data.velocity_magnitude = normalize_speed(flight_data.velocity_magnitude)
    flight_data.horizontal_speed = normalize_speed(flight_data.horizontal_speed)
    flight_data.vertical_speed = normalize_speed(flight_data.vertical_speed)

    # Angles
    flight_data.pitch = normalize_angle(flight_data.pitch)
    flight_data.heading = normalize_angle(flight_data.heading)
    flight_data.roll = normalize_angle(flight_data.roll)
    flight_data.aoa = normalize_angle(flight_data.aoa)

    # Velocity Vector
    x: float = flight_data.velocity_x
    y: float = flight_data.velocity_y
    z: float = flight_data.velocity_z
    unit_vector: tuple[float, float, float] = velocity_unit_vector((x, y, z))
    flight_data.velocity_x = unit_vector[0]
    flight_data.velocity_y = unit_vector[1]
    flight_data.velocity_z = unit_vector[2]

    return flight_data


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

        # do nothing
        case _:
            pass

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


def determine_speed_reward(client: Client) -> float:
    """
    Calculates a reward value based on the thrust-to-weight ratio of the active vessel.

    If the thrust-to-weight ratio is outside the range defined by `MINIMUM_TWR` and `MAXIMUM_TWR`,
    the function returns a negative reward. Otherwise, a positive reward is returned.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The reward value.
    """
    active_vessel = client.space_center.active_vessel
    thrust: float = active_vessel.available_thrust
    weight: float = max(0.01, active_vessel.mass)
    twr: float = thrust / weight

    if (twr < MINIMUM_TWR) or (twr > MAXIMUM_TWR):
        return NEGATIVE_REWARD
    return POSITIVE_REWARD


def determine_altitude_reward(client: Client) -> float:
    """
    Determines the reward based on the altitude of the active vessel.

    If the altitude is outside the range defined by `MINIMUM_ALTITUDE` and `MAXIMUM_ALTITUDE`,
    the function returns a negative reward. Otherwise, a positive reward is returned.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The reward value.
    """
    active_vessel = client.space_center.active_vessel
    altitude: float = active_vessel.flight().mean_altitude

    if (altitude < MINIMUM_ALTITUDE) or (altitude > MAXIMUM_ALTITUDE):
        return NEGATIVE_REWARD
    return POSITIVE_REWARD


def determine_stall_reward(client: Client) -> float:
    """
    Calculates a reward value based on whether the active vessel is stalling or not.

    If the stall ratio is greater than the `MAXIMUM_STALL_RATIO`, the function returns
    a negative reward. Otherwise, a positive reward is returned.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The reward value.
    """
    active_vessel = client.space_center.active_vessel
    vessel_mass: float = active_vessel.mass
    vessel_newtons: float = vessel_mass * client.space_center.g
    lift_vector: tuple[float, float, float] = active_vessel.flight().lift
    lift_magnitude: float = max(0.001, math.sqrt(
        lift_vector[0] ** 2 + lift_vector[1] ** 2 + lift_vector[2] ** 2
    ))
    stall_ratio: float = vessel_newtons / lift_magnitude

    if stall_ratio > MAXIMUM_STALL_RATIO:
        return NEGATIVE_REWARD
    return POSITIVE_REWARD


def determine_position_reward(
    target_point: FlightPoint,
    old_flight_data: NDArray[np.float32],
    new_flight_data: NDArray[np.float32],
) -> float:
    """
    Calculates a reward value based on the change in distance between the old and new
    positions of the active vessel and the target point.

    Args:
        target_point (FlightPoint): The target point to which the aircraft is trying to fly.
        old_flight_data (NDArray[np.float32]): The previous flight data,
            including position and altitude.
        new_flight_data (NDArray[np.float32]): The new flight data,
            including position and altitude.

    Returns:
        float: The reward value.
    """
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
    return NEGATIVE_REWARD


def determine_gforce_reward(client: Client) -> float:
    """
    Calculates a reward value based on the g-force experienced by the active vessel.

    If the absolute value of the g-force exceeds `MAXIMUM_GFORCE`,
    the function returns a negative reward. Otherwise, a positive reward is returned.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The reward value.
    """

    active_vessel = client.space_center.active_vessel
    gforce = active_vessel.flight().g_force

    if abs(gforce) > MAXIMUM_GFORCE:
        return NEGATIVE_REWARD
    return POSITIVE_REWARD


def determine_aoa_reward(client: Client) -> float:
    """
    Calculates a reward value based on the angle of attack of the active vessel.

    If the absolute value of the angle of attack is greater than `MAXIMUM_AOA`,
    the function returns a negative reward. Otherwise, a positive reward is returned.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The reward value.
    """
    active_vessel = client.space_center.active_vessel
    aoa = active_vessel.flight().angle_of_attack

    if abs(aoa) > MAXIMUM_AOA:
        return NEGATIVE_REWARD
    return POSITIVE_REWARD


def determine_roll_angle_reward(client: Client) -> float:
    """
    Calculates a reward value based on the roll angle of the active vessel.

    If the roll angle is outside the range defined by `MAXIMUM_ROLL_ANGLE`,
    the function returns a negative reward. Otherwise, a positive reward is returned.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The reward value.
    """
    active_vessel = client.space_center.active_vessel
    roll_angle: float = active_vessel.flight().roll

    normalized_roll_angle = normalize_angle(roll_angle)
    normalized_max_roll_angle = normalize_angle(MAXIMUM_ROLL_ANGLE)

    if abs(normalized_roll_angle) > normalized_max_roll_angle:
        return NEGATIVE_REWARD
    return POSITIVE_REWARD


def determine_pitch_angle_reward(client: Client) -> float:
    """
    Calculates a reward value based on the pitch angle of the active vessel.

    If the pitch angle is outside the range defined by `MAXIMUM_PITCH_ANGLE`,
    the function returns a negative reward. Otherwise, a positive reward is returned.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The reward value.
    """
    active_vessel = client.space_center.active_vessel
    pitch_angle: float = active_vessel.flight().pitch

    normalized_pitch_angle = normalize_angle(pitch_angle)
    normalized_max_pitch_angle = normalize_angle(MAXIMUM_PITCH_ANGLE)

    if abs(normalized_pitch_angle) > normalized_max_pitch_angle:
        return NEGATIVE_REWARD
    return POSITIVE_REWARD


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


def determine_reward(
    client: Client,
    target_point: FlightPoint,
    old_flight_data: NDArray[np.float32],
    new_flight_data: NDArray[np.float32],
) -> float:
    if has_crashed(client=client):
        return CRASH_REWARD

    speed_reward: float = determine_speed_reward(client=client)
    altitude_reward: float = determine_altitude_reward(client=client)
    position_reward: float = determine_position_reward(
        target_point=target_point,
        old_flight_data=old_flight_data,
        new_flight_data=new_flight_data,
    )
    gforce_reward: float = determine_gforce_reward(client=client)
    aoa_reward: float = determine_aoa_reward(client=client)
    roll_angle_reward: float = determine_roll_angle_reward(client=client)
    stall_reward: float = determine_stall_reward(client=client)
    pitch_angle_reward: float = determine_pitch_angle_reward(client=client)

    return (
        (W_SPEED * speed_reward)
        + (W_ALTITUDE * altitude_reward)
        + (W_POSITION * position_reward)
        + (W_GFORCE * gforce_reward)
        + (W_AOA * aoa_reward)
        + (W_ROLL_ANGLE * roll_angle_reward)
        + (W_STALL * stall_reward)
        + (W_PITCH_ANGLE * pitch_angle_reward)
    )


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
    if has_crashed(client=client) or check_if_arrived(client=client):
        is_terminal = True

    return new_flight_data, reward, is_terminal


def check_if_arrived(client: Client) -> bool:
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

    flight_data: FlightDataPacket = get_flight_data(client=client)
    normalized_flight_data: FlightDataPacket = normalize_flight_data(flight_data)
    flight_data_vector: list[float] = normalized_flight_data.to_vector()
    flight_data_vector.append(target_point.latitude)
    flight_data_vector.append(target_point.longitude)
    flight_data_vector.append(normalize_altitude(target_point.altitude))

    return np.array(flight_data_vector, dtype=np.float32)


def start_over(client: Client, flight_plan: FlightPlan) -> None:
    """
    Starts the game over from the beginning.

    Args:
        client (Client): The kRPC client used to interact with the game.
        flight_plan (FlightPlan): The flight plan containing the route set of flight points.
    """
    print("Starting over...")
    client.space_center.quickload()
    for second in range(5):
        time.sleep(1)
        print(f"Starting over in {5 - second} seconds...")
    execute_flight_plan(client=client, flight_plan=flight_plan)


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
        observation: NDArray[np.float32] = make_observation(
            client=client, target_point=flight_point
        )

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

            # Check if crashed
            if has_crashed(client=client):
                is_done = True
                break

            # Update the observation
            observation = new_observation

            # Check training progress
            reward_buffer.append(reward)
            if len(reward_buffer) % 500 == 0:
                print(f"Latest Observation: {observation}")
                print(f"Average reward: {sum(reward_buffer) / len(reward_buffer)}")
                reward_buffer = []
                agent.save()

        if is_done:
            break

        # Print progress
        if 100 % fp_index == 0:
            print(
                f"Completed {fp_index} flight points out of {len(flight_plan.route_set)}"
            )

    # Start over
    start_over(client=client, flight_plan=flight_plan)
