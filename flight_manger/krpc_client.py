"""
This module provides functions for interacting with the kRPC client and
manipulating the flight controls of the active vessel.
"""

import krpc
from krpc.client import Client
from krpc.services.spacecenter import Control, WaypointManager

from flight_manger.data import Coordinate

DEFAULT_ADDRESS: str = "172.31.160.1"
DEFAULT_MARGINAL: float = 0.01
CONTROL_SCHEMA: dict[int, str] = {
    0: "throttle up",
    1: "throttle down",
    2: "pitch up",
    3: "pitch down",
    4: "yaw left",
    5: "yaw right",
    6: "roll left",
    7: "roll right",
}


def get_client() -> Client:
    """
    Establishes a connection to the kRPC server and returns a Client object.

    Returns:
        Client: A kRPC Client object connected to the "Flight Manager" server.
    """

    return krpc.connect(name="Flight Manager", address=DEFAULT_ADDRESS)


def create_waypoint(client: Client, coordinate: Coordinate) -> None:
    """
    Creates a waypoint at the specified geographic coordinate.

    This function adds a waypoint to the active vessel's celestial body
    using the kRPC client's waypoint manager. The waypoint is given a
    name based on the provided coordinate.

    Args:
        client (Client): The kRPC client used to interact with the game.
        coordinate (Coordinate): The geographic coordinate where the waypoint
            will be created.

    Returns:
        None
    """

    waypoint_manager: WaypointManager = client.space_center.waypoint_manager
    waypoint_manager.add_waypoint(
        latitude=coordinate.latitude,
        longitude=coordinate.longitude,
        body=client.space_center.active_vessel.orbit.body,
        name=f"Target {coordinate}",
    )


def delete_waypoints(client: Client) -> None:
    """
    Removes all waypoints from the active vessel's waypoint manager.

    This function iterates over the list of waypoints in the waypoint manager and
    removes each one. This is useful for clearing the waypoints from the previous
    iteration of the reinforcement learning algorithm.

    Args:
        client (Client): The kRPC client from which the waypoint manager is retrieved.
    """
    for waypoint in client.space_center.waypoint_manager.waypoints:
        waypoint.remove()


def manipulate_controls(client: Client, action_id: int) -> None:
    """
    Adjusts the flight controls of the active vessel based on the given action ID.

    The adjustments are made in the following way:
        0: Throttle up by DEFAULT_MARGINAL
        1: Throttle down by DEFAULT_MARGINAL
        2: Pitch up by DEFAULT_MARGINAL
        3: Pitch down by DEFAULT_MARGINAL
        4: Yaw left by DEFAULT_MARGINAL
        5: Yaw right by DEFAULT_MARGINAL
        6: Roll left by DEFAULT_MARGINAL
        7: Roll right by DEFAULT_MARGINAL

    Args:
        client (Client): The kRPC client from which the active vessel is retrieved.
        action_id (int): An integer representing the action to be taken. The actions are:
            0: Throttle up
            1: Throttle down
            2: Pitch up
            3: Pitch down
            4: Yaw left
            5: Yaw right
            6: Roll left
            7: Roll right
    """
    control_manager: Control = client.space_center.active_vessel.control

    match action_id:
        # Throttle up
        case 0:
            control_manager.throttle += DEFAULT_MARGINAL

        # Throttle down
        case 1:
            control_manager.throttle -= DEFAULT_MARGINAL

        # Pitch up
        case 2:
            control_manager.pitch += DEFAULT_MARGINAL

        # Pitch down
        case 3:
            control_manager.pitch -= DEFAULT_MARGINAL

        # Yaw left
        case 4:
            control_manager.yaw += DEFAULT_MARGINAL

        # Yaw right
        case 5:
            control_manager.yaw -= DEFAULT_MARGINAL

        # Roll left
        case 6:
            control_manager.roll += DEFAULT_MARGINAL

        # Roll right
        case 7:
            control_manager.roll -= DEFAULT_MARGINAL

        # Do nothing
        case 8:
            pass

        # Invalid action ID
        case _:
            raise ValueError(f"Invalid action ID: {action_id}")
