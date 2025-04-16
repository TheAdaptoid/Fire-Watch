"""
This module contains the Observation class for storing and processing flight data.
It also includes functions for normalizing and converting observations to NumPy ndarrays.
"""

from dataclasses import dataclass

import numpy as np
import torch
from krpc.client import Client
from numpy.typing import NDArray
from torch.types import Tensor

from flight_manger.cost_functions import W_HEADING, heading_reward
from flight_manger.utils import Coordinate, negative_to_positive, zero_to_one

ALTITUDE_ASL_MIN: float = 0
ALTITUDE_ASL_MAX: float = 12000  # 40K ft ceiling
PITCH_ANGLE_MIN: float = -90
PITCH_ANGLE_MAX: float = 90
ROLL_ANGLE_MIN: float = -180
ROLL_ANGLE_MAX: float = 180
HEADING_DIFFERENCE_MIN: float = -180
HEADING_DIFFERENCE_MAX: float = 180


@dataclass
class Observation:
    """Observation class for storing and processing flight data."""

    altitude_asl: float
    pitch_angle: float
    roll_angle: float
    heading_offset: float

    def normalize_values(self) -> None:
        """
        Normalize the values of the Observation instance to a range of [0, 1].

        This function normalizes the `altitude_asl`, `pitch_angle`, `roll_angle` and
        `heading_offset` attributes of the Observation instance to a range of [0, 1].
        The normalization is done by shifting the range of each attribute to [0, 1]
        using the corresponding min and max values.

        The normalized values are stored in the same attributes, overwriting the
        original values.
        """
        self.altitude_asl = zero_to_one(
            self.altitude_asl, ALTITUDE_ASL_MIN, ALTITUDE_ASL_MAX
        )
        self.pitch_angle = negative_to_positive(
            self.pitch_angle, PITCH_ANGLE_MIN, PITCH_ANGLE_MAX
        )
        self.roll_angle = negative_to_positive(
            self.roll_angle, ROLL_ANGLE_MIN, ROLL_ANGLE_MAX
        )
        self.heading_offset = negative_to_positive(self.heading_offset, -1, 1)

    def to_tensor(self, normalize: bool = False) -> Tensor:
        """
        Converts the observation to a PyTorch tensor of floats.

        If normalize is True, the observation values are normalized before being
        converted to a tensor.

        Args:
            normalize (bool, optional): If True, normalize the observation values.
                Defaults to False.

        Returns:
            Tensor: A PyTorch tensor of floats representing the observation.
        """
        if normalize:
            self.normalize_values()

        return torch.tensor(
            [
                self.altitude_asl,
                self.pitch_angle,
                self.roll_angle,
                self.heading_offset,
            ],
            dtype=torch.float32,
        )

    def to_ndarray(self, normalize: bool = False) -> NDArray[np.float32]:
        """
        Converts the observation to a NumPy ndarray of floats.

        If normalize is True, the observation values are normalized before being
        converted to an ndarray.

        Args:
            normalize (bool, optional): If True, normalize the observation values.
                Defaults to False.

        Returns:
            NDArray[np.float32]: A NumPy ndarray of floats representing the observation.
        """
        if normalize:
            self.normalize_values()

        return np.array(
            [
                self.altitude_asl,
                self.pitch_angle,
                self.roll_angle,
                self.heading_offset,
            ],
            dtype=np.float32,
        )

    def to_tuple(self, normalize: bool = False) -> tuple[float, ...]:
        """
        Returns a tuple of floats representing the observation.

        If normalize is True, the values are normalized to the ranges [0, 1] and [-1, 1] before
        being returned.

        Returns:
            tuple[float, ...]: A tuple of floats representing the observation.
        """
        if normalize:
            self.normalize_values()

        return (
            self.altitude_asl,
            self.pitch_angle,
            self.roll_angle,
            self.heading_offset,
        )

    @property
    def length(self) -> int:
        """
        Returns the length of the observation when represented as a tuple.

        This property calculates the length of the tuple representation of the
        observation, providing the number of elements it contains.

        Returns:
            int: The number of elements in the observation tuple.
        """

        return len(self.to_tuple())


def observe(client: Client, target: Coordinate) -> Observation:
    """
    Observes the current state of the vessel and returns an Observation object.

    The returned observation contains the current altitude above sea level,
    pitch angle, roll angle, and the difference between the vessel's heading and
    the target direction.

    Args:
        client (Client): The kRPC client used to interface with the game.
        target (Coordinate): The target coordinate for flight navigation.

    Returns:
        Observation: An Observation object containing the vessel's state.
    """

    vessel = client.space_center.active_vessel
    flight_data = vessel.flight()

    observation = Observation(
        altitude_asl=flight_data.mean_altitude,
        pitch_angle=flight_data.pitch,
        roll_angle=flight_data.roll,
        heading_offset=heading_reward(client, target) / W_HEADING,
    )

    return observation
