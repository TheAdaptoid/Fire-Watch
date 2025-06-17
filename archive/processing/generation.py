import numpy as np
from noise import pnoise2

from flight_procedures.utils import HotSpot, timed_function

def map_to_range(
    x: float, new_min: float, new_max: float, old_min: float, old_max: float
) -> float:
    """
    Maps a value from one range to another.
    This function maps a value, x, from the range [xMin, xMax] to the range [a, b].

    Args:
        x (float): The value to map.
        new_min (float): The minimum value of the new range.
        new_max (float): The maximum value of the new range.
        old_min (float): The minimum value of the old range.
        old_max (float): The maximum value of the old range.

    Returns:
        float: The mapped value.
    """
    return new_min + (new_max - new_min) * (x - old_min) / (old_max - old_min)


def normalize_noise(noise: np.ndarray) -> np.ndarray:
    """
    Normalizes a noise matrix to the range [0, 1].

    Args:
        noise (np.ndarray): The noise matrix to normalize.

    Returns:
        np.ndarray: The normalized noise matrix.
    """
    return (noise - noise.min()) / (noise.max() - noise.min())


def generate_noise_map(
    size: int = 100,
    seed: int = -1,
    octaves: int = 3,
    persistence: float = 0.75,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """
    Generates a perlin noise map.

    Args:
        size (int, optional): The size of the noise map.
            The noise map will contain `size`^2 points. Defaults to 100.
        seed (int, optional): The seed to use to generate the noise. Defaults to -1.
        octaves (int, optional): The number of octaves to use.
            Affects the detail of the noise. Defaults to 3.
        persistence (float, optional): The persistence of the noise.
            Affects the roughness of the noise. Defaults to 0.75.
        lacunarity (float, optional): The lacunarity of the noise.
            Affects the frequency of the noise. Defaults to 2.0.

    Returns:
        np.ndarray: The noise map.
    """
    if seed < 0:
        seed = np.random.randint(0, 1000000)

    noise = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            noise[i][j] = pnoise2(
                i / size,
                j / size,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                base=seed,
            )

    return normalize_noise(noise)


def apply_cutoff(noise: np.ndarray, cutoff: float) -> np.ndarray:
    """
    Applies a cutoff to a noise matrix.

    Args:
        noise (np.ndarray): The noise matrix to apply the cutoff to.
        cutoff (float): The cutoff value.

    Returns:
        np.ndarray: The noise matrix with the cutoff applied.
    """
    return np.where(noise < cutoff, 0.0, noise)


def convert_to_coordinates(noise: np.ndarray) -> list[tuple[int, int]]:
    """
    Converts a noise matrix to a list of coordinates.

    Args:
        noise (np.ndarray): The noise matrix to convert.

    Returns:
        list[tuple[int, int]]: A list of tuples containing the coordinates
            of the non-zero elements in the noise matrix.
    """
    coordinates: list[tuple[int, int]] = []

    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            if noise[i][j] != 0:
                coordinates.append((i, j))

    return coordinates


def adjust_axis_ranges(
    coordinates: list[tuple[int, int]],
    longitude_range: tuple[float, float],
    latitude_range: tuple[float, float],
    x_length: int,
    y_length: int,
) -> list[tuple[float, float]]:
    """
    Adjusts the x and y axis ranges to match the desired longitude and latitude ranges.

    Args:
        coordinates (list[tuple[int, int]]): The coordinates to adjust.
        longitude_range (tuple[float, float]): The desired longitude range.
        latitude_range (tuple[float, float]): The desired latitude range.
        x_length (int): The length of the x axis.
        y_length (int): The length of the y axis.

    Returns:
        list[tuple[float, float]]: The adjusted coordinates.
    """
    adjusted_coordinates: list[tuple[float, float]] = []

    for coordinate in coordinates:
        adjusted_coordinates.append(
            (
                map_to_range(
                    x=coordinate[0],
                    new_min=longitude_range[0],
                    new_max=longitude_range[1],
                    old_min=0,
                    old_max=x_length,
                ),
                map_to_range(
                    x=coordinate[1],
                    new_min=latitude_range[0],
                    new_max=latitude_range[1],
                    old_min=0,
                    old_max=y_length,
                ),
            )
        )

    return adjusted_coordinates

@timed_function
def generate_hot_spots(
    latitude_range: tuple[float, float],
    longitude_range: tuple[float, float],
    size: int = 1000,
    seed: int = 42,
) -> list[HotSpot]:
    """
    Creates a list of HotSpot objects.

    Args:
        latitude_range (tuple[float, float]): A tuple containing the minimum
            and maximum latitude values.
            Index 0 is the minimum and 1 is the maximum.
        longitude_range (tuple[float, float]): A tuple containing the minimum
            and maximum longitude values.
            Index 0 is the minimum and 1 is the maximum.
        size (int, optional): The size of the noise map.
            The noise map will contain `size`^2 points.
            Defaults to 1000.
        seed (int, optional): The random seed used to generate the noise. Defaults to 42.

    Returns:
        list[HotSpot]: A list of HotSpot objects.
    """

    noise_map: np.ndarray = generate_noise_map(size=size, seed=seed)
    noise_map_truncated: np.ndarray = apply_cutoff(noise_map, 0.75)
    coordinates: list[tuple[int, int]] = convert_to_coordinates(noise_map_truncated)
    adjusted_coordinates: list[tuple[float, float]] = adjust_axis_ranges(
        coordinates=coordinates,
        longitude_range=longitude_range,
        latitude_range=latitude_range,
        x_length=size,
        y_length=size,
    )

    return list(
        map(
            lambda coordinate: HotSpot(
                lattitude=coordinate[0], longitude=coordinate[1]
            ),
            adjusted_coordinates,
        )
    )
