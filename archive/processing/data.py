from krpc.client import Client
from numpy.typing import NDArray
import numpy as np

from flight_procedures.processing.normals import MACH_SPEED
from flight_procedures.utils import Coordinate

MAX_ALTITUDE: int = 10000
MIN_ALTITUDE: int = 1000

MACH_SPEED: float = 340.3


def normalize(value: float, min_value: float, max_value: float) -> float:
    numerator: float = value - min_value
    denominator: float = max_value - min_value

    if denominator == 0:
        denominator = 0.001

    return numerator / denominator


def get_altitude_agl_ratio(client: Client) -> float:
    """
    Calculates the normalized altitude above ground level (AGL) for the active vessel.

    The altitude is normalized between `MIN_ALTITUDE` and `MAX_ALTITUDE` to provide
    a ratio. This ratio can be used for further analysis or control decisions.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The normalized altitude AGL ratio.
    """

    active_vessel = client.space_center.active_vessel
    return normalize(
        active_vessel.flight().surface_altitude, MIN_ALTITUDE, MAX_ALTITUDE
    )


def get_altitude_asl_ratio(client: Client) -> float:
    """
    Calculates the normalized altitude above sea level (ASL) for the active vessel.

    The altitude is normalized between `MIN_ALTITUDE` and `MAX_ALTITUDE` to provide
    a ratio. This ratio can be used for further analysis or control decisions.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The normalized altitude ASL ratio.
    """
    active_vessel = client.space_center.active_vessel
    return normalize(active_vessel.flight().mean_altitude, MIN_ALTITUDE, MAX_ALTITUDE)


def get_true_airspeed_ratio(client: Client) -> float:
    """
    Calculates the ratio of true airspeed to Mach speed for the active vessel.

    This function retrieves the true airspeed of the active vessel and normalizes
    it by dividing by the speed of sound (Mach speed). The result is a dimensionless
    ratio that can be used for further analysis or control decisions.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The true airspeed ratio as a dimensionless number.
    """

    active_vessel = client.space_center.active_vessel
    return active_vessel.flight().true_air_speed / MACH_SPEED


def get_effective_airspeed_ratio(client: Client) -> float:
    """
    Calculates the ratio of effective airspeed to Mach speed for the active vessel.

    This function retrieves the effective airspeed of the active vessel and normalizes
    it by dividing by the speed of sound (Mach speed). The result is a dimensionless
    ratio that can be used for further analysis or control decisions.

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.

    Returns:
        float: The effective airspeed ratio as a dimensionless number.
    """

    active_vessel = client.space_center.active_vessel
    return active_vessel.flight().equivalent_air_speed / MACH_SPEED


def get_pitch_percentage(client: Client) -> float:
    active_vessel = client.space_center.active_vessel
    return normalize(active_vessel.flight().pitch, -90, 90)


def get_roll_percentage(client: Client) -> float:
    active_vessel = client.space_center.active_vessel
    return normalize(active_vessel.flight().roll, -180, 180)


def get_heading_angle(client: Client) -> float:
    active_vessel = client.space_center.active_vessel
    return active_vessel.flight().heading


def get_throttle_percentage(client: Client) -> float:
    active_vessel = client.space_center.active_vessel
    return normalize(active_vessel.control.throttle, 0, 1)


def get_thrust_ratio(client: Client) -> float:
    active_vessel = client.space_center.active_vessel
    return active_vessel.thrust / max(active_vessel.available_thrust, 0.01)


def get_intake_flow_ratio(client: Client) -> float:
    active_vessel = client.space_center.active_vessel

    total_intake_speed: float = 0.0  # Meters per second
    total_intake_size: float = 0.0  # Square meters
    intake_count: int = 0
    for index, intake in enumerate(active_vessel.parts.intakes):
        total_intake_speed += intake.speed
        total_intake_size += intake.area
        intake_count += 1

    if intake_count == 0:
        return 0

    return (total_intake_speed / intake_count) / max(
        (total_intake_size / intake_count), 0.01
    )


def observe_environment(client: Client, destination: Coordinate) -> NDArray[np.float32]:

    observation: NDArray[np.float32] = np.array(
        [
            # ASL
            get_altitude_asl_ratio(client=client),
            # AGL
            get_altitude_agl_ratio(client=client),
            # True Airspeed
            get_true_airspeed_ratio(client=client),
            # Effective Airspeed
            get_effective_airspeed_ratio(client=client),
            # Roll Angle
            get_roll_percentage(client=client),
            # Pitch Angle
            get_pitch_percentage(client=client),
            # Heading Angle
            get_heading_angle(client=client),
            # Throttle
            get_throttle_percentage(client=client),
            # Thrust
            get_thrust_ratio(client=client),
            # Intake Flow Ratio
            get_intake_flow_ratio(client=client),
        ],
        dtype=np.float32,
    )

    # print(f"Observation: {observation}")

    return observation
