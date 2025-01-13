import numpy as np
from matplotlib import pyplot as plt
from noise import pnoise2
from os import system

LONGITUDE_RANGE: tuple[float, float] = (-78, -73)
LATITUDE_RANGE: tuple[float, float] = (0, 5)
OCTAVES: int = 3
PERSISTENCE: float = 0.75
LACUNARITY: float = 2.0

def Map_To_Range(a: float, b: float, x: float, xMin: float, xMax: float) -> float:
    return a + (b - a) * (x - xMin) / (xMax - xMin)

def Generate_Fire_Map(seed: int = 0, precision: int = 100) -> list[tuple[int, int]]:
    """
    Generate a fire map based on a seed.

    This function generates a fire map, represented as a list of (longitude, latitude) pairs, based on a given seed. The generated map is normalized to fit within the given longitude and latitude ranges.

    Parameters
    ----------
    seed : int, optional
        The seed for the noise generation. Defaults to 0.

    Returns
    -------
    list[tuple[int, int]]
        A list of (longitude, latitude) pairs representing the fire map.
    """

    # Create a noise mattrix
    noise = np.zeros((precision, precision))
    for i in range(precision):
        for j in range(precision):
            noise[i][j] = pnoise2(i/precision, j/precision, octaves=OCTAVES, persistence=PERSISTENCE, lacunarity=LACUNARITY, base=seed)

    # Normalize
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = np.where(noise < 0.75, 0.0, noise)

    # Create a fire map
    fireMap: list[tuple[int, int]] = []

    # Convert noise to fire map
    for i in range(precision):
        for j in range(precision):
                if noise[i][j] != 0:
                    fireMap.append((
                        Map_To_Range(LONGITUDE_RANGE[0], LONGITUDE_RANGE[1], i, 0, precision),
                        Map_To_Range(LATITUDE_RANGE[0], LATITUDE_RANGE[1], j, 0, precision),
                    ))

    return fireMap