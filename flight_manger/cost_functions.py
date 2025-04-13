"""
This module contains the cost functions used in the reinforcement learning
algorithm to guide the vessel's actions.
"""

import numpy as np
from krpc.client import Client
from krpc.services.spacecenter import Flight, VesselSituation

from flight_manger.utils import (
    Coordinate,
    absolute_negation,
    create_logger,
    heading_to_direction_vector,
    negative_to_positive,
    target_to_direction_vector,
    zero_to_one,
)

IDEAL_ALTITUDE: int = 6000  # (meters) Best for the engines in the simulation

# Weight constants
W_PITCH: float = 10
W_ROLL: float = 10
W_ALTITUDE: float = 12
W_HEADING: float = 2


def __pitch_reward(pitch: float) -> float:
    """
    Computes the reward for the pitch angle of a vessel.

    The function processes the pitch angle by mapping it from its original range
    of [-90, 90] to [-1, 1], and then inverses the mapped value to fit the reward
    structure where an ideal pitch (0 degrees) results in the highest reward.

    Args:
        pitch (float): The current pitch angle of the vessel in degrees.

    Returns:
        float: The processed and inversed pitch reward, where values closer to 0 degrees
        yield a higher reward.
    """

    processed_pitch: float = negative_to_positive(pitch, -90, 90)
    inversed_pitch: float = absolute_negation(processed_pitch)
    return inversed_pitch


def pitch_reward(client: Client) -> float:
    """
    Computes the reward for the pitch angle of a vessel.

    The function processes the pitch angle by mapping it from its original range
    of [-90, 90] to [-1, 1], and then inverses the mapped value to fit the reward
    structure where an ideal pitch (0 degrees) results in the highest reward.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The processed and inversed pitch reward, where values closer to 0 degrees
        yield a higher reward.
    """
    flight_data: Flight = client.space_center.active_vessel.flight()
    raw_pitch: float = flight_data.pitch
    reward: float = __pitch_reward(raw_pitch)
    return W_PITCH * reward


def __roll_reward(roll: float) -> float:
    """
    Processes the roll angle of a vessel by mapping it from its original range
    of [-180, 180] to [-1, 1], and then inverses the mapped value to fit the reward
    structure where an ideal roll (0 degrees) results in the highest reward.

    Args:
        roll (float): The current roll angle of the vessel in degrees.

    Returns:
        float: The processed and inversed roll reward, where values closer to 0 degrees
        yield a higher reward.
    """
    processed_roll: float = negative_to_positive(roll, -180, 180)
    inversed_roll: float = absolute_negation(processed_roll)
    return inversed_roll


def roll_reward(client: Client) -> float:
    """
    Computes the reward for the roll angle of a vessel.

    The function processes the roll angle by mapping it from its original range
    of [-180, 180] to [-1, 1], and then inverses the mapped value to fit the reward
    structure where an ideal roll (0 degrees) results in the highest reward.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The processed and inversed roll reward, where values closer to 0 degrees
        yield a higher reward.
    """
    flight_data: Flight = client.space_center.active_vessel.flight()
    raw_roll: float = flight_data.roll
    reward: float = __roll_reward(raw_roll)
    return W_ROLL * reward


def __altitude_reward(altitude: float) -> float:
    """
    Computes the reward for the altitude of a vessel.

    The function calculates the absolute difference between the current altitude
    and an ideal altitude, then normalizes this difference to a range [0, 1],
    and inverses the normalized value to fit the reward structure where an ideal
    altitude results in the highest reward.

    Args:
        altitude (float): The current altitude of the vessel in meters.

    Returns:
        float: The processed and inversed altitude reward, where values closer
        to the ideal altitude yield a higher reward.
    """

    absolute_difference: float = abs(altitude - IDEAL_ALTITUDE)
    normalized_valued: float = zero_to_one(absolute_difference, 0, IDEAL_ALTITUDE)
    inversed_valued: float = absolute_negation(normalized_valued)
    return inversed_valued


def altitude_reward(client: Client) -> float:
    """
    Computes the reward for the altitude of a vessel.

    The function calculates the absolute difference between the current altitude
    and an ideal altitude, then normalizes this difference to a range [0, 1],
    and inverses the normalized value to fit the reward structure where an ideal
    altitude results in the highest reward.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The processed and inversed altitude reward, where values closer
        to the ideal altitude yield a higher reward.
    """
    flight_data: Flight = client.space_center.active_vessel.flight()
    raw_altitude: float = flight_data.mean_altitude
    reward: float = __altitude_reward(raw_altitude)
    return W_ALTITUDE * reward


def heading_reward(client: Client, target: Coordinate) -> float:
    """
    Computes the reward for the heading angle of a vessel.

    The function calculates the alignment score between the current heading of the
    vessel and the direction vector from the vessel to the target coordinate. The
    alignment score is calculated as the dot product of the normalized heading vector
    and the normalized target direction vector. The result is then multiplied by the
    weight for the heading reward to obtain the final reward.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.
        target (Coordinate): The target coordinate for which the heading reward is
            calculated.

    Returns:
        float: The heading reward, where values closer to 1 indicate better alignment
            between the current heading and the target direction.
    """
    kerbin_radius: float = (
        client.space_center.active_vessel.orbit.body.equatorial_radius
    )
    flight_data: Flight = client.space_center.active_vessel.flight()

    # Calculate heading direction vector
    heading_vector: tuple[float, float] = heading_to_direction_vector(
        flight_data.heading
    )

    # Calculate target direction vector
    c_lat: float = flight_data.latitude
    c_lon: float = flight_data.longitude
    target_vector: tuple[float, float] = target_to_direction_vector(
        target=target.to_tuple(), current=(c_lat, c_lon), kerbin_radius=kerbin_radius
    )

    # Calculate alignment score
    hv_normalized: tuple[np.floating, np.floating] = (
        heading_vector[0] / np.linalg.norm(heading_vector),
        heading_vector[1] / np.linalg.norm(heading_vector),
    )
    tv_normalized: tuple[np.floating, np.floating] = (
        target_vector[0] / np.linalg.norm(target_vector),
        target_vector[1] / np.linalg.norm(target_vector),
    )
    dot_product: float = np.dot(hv_normalized, tv_normalized)

    return W_HEADING * dot_product


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
    situation: VesselSituation = client.space_center.active_vessel.situation
    return situation != VesselSituation.flying


def has_arrived(client: Client, target: Coordinate) -> bool:
    """
    Determines if the active vessel has reached its target destination.

    This function checks whether the vessel's current position aligns with
    the target destination, indicating a successful arrival.

    Args:
        client (Client): The kRPC client used to interact with the game.
        target (Coordinate): The target coordinate for flight navigation.

    Returns:
        bool: True if the vessel has arrived at the target destination, False otherwise.
    """
    # TODO: implement
    return False


def calculate_reward(client: Client, target: Coordinate) -> tuple[float, bool]:
    """
    Calculates the total reward for the active vessel based on several factors
    and determines if the flight episode is complete.

    This function computes individual rewards for the vessel's pitch, roll,
    altitude, and heading, and sums them to form a total reward. It also checks
    if the vessel has either crashed or arrived at the target destination to
    adjust the reward and determine if the episode is done.

    Args:
        client (Client): The kRPC client used to interact with the game.
        target (Coordinate): The target coordinate for flight navigation.

    Returns:
        tuple[float, bool]: A tuple containing the total reward and a boolean
        indicating if the episode is complete. A higher reward indicates better
        adherence to ideal flight parameters.
    """

    reward: list[float] = [
        pitch_reward(client=client),
        roll_reward(client=client),
        altitude_reward(client=client),
        heading_reward(client=client, target=target),
    ]
    reward_total: float = sum(reward)

    is_done: bool = False
    if has_crashed(client=client):
        reward_total = reward_total - 100
        is_done = True
    elif has_arrived(client=client, target=target):
        reward_total = reward_total + 100
        is_done = True

    # Write logs
    logger = create_logger()
    logger.debug(
        "Done %s | Reward Total: %s | Breakdown: %s", is_done, reward_total, reward
    )

    return reward_total, is_done
