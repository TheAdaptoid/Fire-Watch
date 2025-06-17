from krpc.client import Client


from flight_procedures.utils import Coordinate
from flight_procedures.processing.normals import magnitude, normalize

ONE_HOUR: int = 3600  # 1 hour in seconds


IDEAL_ALTITUDE: int = 9144  # 30,000 ft in meters
FLIGHT_FLOOR: int = 5000
FLIGHT_CEILING: int = 10000

# Weights
W_DURATION: float = 0.25
W_AOA: float = 0.25
W_AOS: float = 0.25
W_GFORCE: float = 0.25
W_ALTITUDE: float = 2
W_PITCH: float = 1
W_ROLL: float = 1
W_EFFICIENCY: float = 0.01
W_SPEED: float = 1


def aoa_reward(client: Client) -> float:
    aoa: float = client.space_center.active_vessel.flight().angle_of_attack
    return -1 * abs(aoa)


def aos_reward(client: Client) -> float:
    aos: float = client.space_center.active_vessel.flight().sideslip_angle
    return -1 * abs(aos)


def gforce_reward(client: Client) -> float:
    gforce: float = client.space_center.active_vessel.flight().g_force
    return -1 * gforce


def pitch_reward(client: Client) -> float:
    pitch: float = client.space_center.active_vessel.flight().pitch
    return -1 * (abs(pitch) / 90)  # Ideal pitch is 0 degrees


def roll_reward(client: Client) -> float:
    roll: float = client.space_center.active_vessel.flight().roll
    return -1 * (abs(roll) / 180)  # Ideal roll is 0 degrees


def altitude_reward(client: Client) -> float:
    altitude: float = client.space_center.active_vessel.flight().mean_altitude
    if altitude < FLIGHT_FLOOR:
        return -1
    if altitude > FLIGHT_CEILING:
        return -1
    return -1 * normalize(abs(IDEAL_ALTITUDE - altitude), 0, IDEAL_ALTITUDE)


def duration_reward(client: Client) -> float:
    mission_time: float = client.space_center.active_vessel.met
    return mission_time / ONE_HOUR


def efficiency_reward(client: Client) -> float:
    active_vessel = client.space_center.active_vessel
    intakes: list = active_vessel.parts.intakes

    total_intake_speed: float = 0.0  # Meters per second
    total_intake_size: float = 0.0  # Square meters
    intake_count: int = 0
    for index, intake in enumerate(intakes):
        total_intake_speed += intake.speed
        total_intake_size += intake.area
        intake_count += 1

    if intake_count == 0:
        return 0

    return (total_intake_speed / intake_count) / max(
        (total_intake_size / intake_count), 0.01
    )

def speed_reward(client: Client) -> float:
    active_vessel = client.space_center.active_vessel
    return active_vessel.flight().mach


def calculate_reward(client: Client, destination: Coordinate) -> float:
    reward_collection: list[float] = [
        (W_DURATION * duration_reward(client=client)),
        (W_AOA * aoa_reward(client=client)),
        (W_AOS * aos_reward(client=client)),
        (W_GFORCE * gforce_reward(client=client)),
        (W_ROLL * roll_reward(client=client)),
        (W_PITCH * pitch_reward(client=client)),
        # (W_EFFICIENCY * efficiency_reward(client=client)),
        (W_ALTITUDE * altitude_reward(client=client)),
        (W_SPEED * speed_reward(client=client)),
    ]

    # print(f"Reward collection: {reward_collection}")

    return sum(reward_collection)
