"""
This module contains utility functions for the flight manager application.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np


def create_logger() -> logging.Logger:
    """
    Creates and configures a logger for the flight manager application.

    The logger is set to the DEBUG level and logs messages to a file named
    "flight_manger.log" in the "logs" directory. Each log entry includes the
    timestamp, log level, and message content.

    Returns:
        logging.Logger: A configured logger instance.
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("logs/flight_manger.log")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


@dataclass
class Coordinate:
    """A class representing a coordinate."""

    latitude: float
    longitude: float

    def to_tuple(self) -> tuple[float, float]:
        """
        Returns a tuple of the latitude and longitude.

        Returns:
            tuple[float, float]: The latitude and longitude as a tuple.
        """
        return (self.latitude, self.longitude)

    def __str__(self) -> str:
        return f"({self.latitude}, {self.longitude})"

    def __repr__(self) -> str:
        return f"Coordinate({self.latitude}, {self.longitude})"


def normalize_value(
    value: int | float, min_value: int | float, max_value: int | float
) -> float:
    """
    Normalize a value to a value between 0 and 1.

    Args:
        value: The value to normalize.
        min_value: The minimum value of the range.
        max_value: The maximum value of the range.

    Returns:
        The normalized value, which is a float between 0 and 1.
    """
    numerator: float = value - min_value
    denominator: float = max_value - min_value

    if denominator == 0:
        denominator = 0.001

    return numerator / denominator


def zero_to_one(
    value: int | float, min_value: int | float, max_value: int | float
) -> float:
    """
    Maps a value to the range [0, 1] based on a given range.

    Args:
        value: The value to be normalized.
        min_value: The minimum value of the original range.
        max_value: The maximum value of the original range.

    Returns:
        A float representing the normalized value within the range [0, 1].
    """

    normalized_value = normalize_value(value, min_value, max_value)
    return normalized_value


def negative_to_positive(
    value: int | float, min_value: int | float, max_value: int | float
) -> float:
    """
    Maps a value from a given range to the range [-1, 1].

    This function works by first normalizing the value to the range [0, 1] and then
    scaling and shifting it to the range [-1, 1].

    Args:
        value: The value to be mapped.
        min_value: The minimum value of the original range.
        max_value: The maximum value of the original range.

    Returns:
        A float representing the mapped value within the range [-1, 1].
    """
    zero_to_one_value = zero_to_one(value, min_value, max_value)
    return zero_to_one_value * 2 - 1


def absolute_negation(value: int | float) -> float:
    """
    Returns the negated absolute value of the input.

    This function calculates the absolute value of the input and then negates it,
    resulting in a non-positive output.

    Args:
        value: An integer or float to be negated.

    Returns:
        A float representing the negated absolute value of the input.
    """

    return -1 * abs(value)


def target_to_direction_vector(
    target: tuple[float, float], current: tuple[float, float], kerbin_radius: float
) -> tuple[float, float]:
    """
    Calculates the direction vector from the current position to the target position
    on a celestial body with a given radius.

    This function computes the direction vector in a 2D plane based on the latitude
    and longitude differences between the target and current positions. The vector
    components are calculated using spherical geometry, considering the curvature
    of the celestial body.

    Args:
        target (tuple[float, float]): The target latitude and longitude in degrees.
        current (tuple[float, float]): The current latitude and longitude in degrees.
        kerbin_radius (float): The radius of the celestial body, such as Kerbin.

    Returns:
        tuple[float, float]: The x and y components of the direction vector.
    """

    # Unpack coordinates
    t_lat_degree: float = target[0]
    t_lon_degree: float = target[1]
    c_lat_degree: float = current[0]
    c_lon_degree: float = current[1]

    # Convert degrees to radians
    t_lat_radians: float = math.radians(t_lat_degree)
    t_lon_radians: float = math.radians(t_lon_degree)
    c_lat_radians: float = math.radians(c_lat_degree)
    c_lon_radians: float = math.radians(c_lon_degree)

    # Calculate deltas
    lat_delta: float = t_lat_radians - c_lat_radians
    lon_delta: float = t_lon_radians - c_lon_radians

    # Calculate vector components
    x_component: float = (
        kerbin_radius * lon_delta * np.cos((t_lat_radians + c_lat_radians) / 2)
    )
    y_component: float = kerbin_radius * lat_delta

    return (x_component, y_component)


def heading_to_direction_vector(heading: float) -> tuple[float, float]:
    """
    Calculates the direction vector for a given heading.

    This function takes a heading in degrees and returns a 2D direction vector
    as a tuple of floats. The direction vector is calculated using the sine
    and cosine of the heading angle, considering the positive x-axis as the
    reference direction (0 degrees).

    Args:
        heading (float): The heading angle in degrees.

    Returns:
        tuple[float, float]: The x and y components of the direction vector.
    """
    # Convert degrees to radians
    heading_radians: float = math.radians(heading)

    # Calculate vector components
    x_component: float = np.sin(heading_radians)
    y_component: float = np.cos(heading_radians)

    return (x_component, y_component)
