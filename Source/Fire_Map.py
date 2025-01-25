import numpy as np
from noise import pnoise2

def Map_To_Range(a: float, b: float, x: float, xMin: float, xMax: float) -> float:
    """
    Maps a value from one range to another.
    This function maps a value, x, from the range [xMin, xMax] to the range [a, b].

    Parameters
    ----------
    a : float
        The lower bound of the target range.
    b : float
        The upper bound of the target range.
    x : float
        The value to map.
    xMin : float
        The lower bound of the original range.
    xMax : float
        The upper bound of the original range.

    Returns
    -------
    float
        The mapped value.
    """
    return a + (b - a) * (x - xMin) / (xMax - xMin)

def Normalize_Noise(noise: np.ndarray) -> np.ndarray:
    """
    Normalize a noise matrix to have values between 0 and 1.

    Parameters
    ----------
    noise : np.ndarray
        The noise matrix to normalize.

    Returns
    -------
    np.ndarray
        The normalized noise matrix, with values ranging from 0 to 1.
    """
    return (noise - noise.min()) / (noise.max() - noise.min())

def Generate_Noise_Map(size: int = 100, seed: int = 0, octaves: int = 3, persistence: float = 0.75, lacunarity: float = 2.0) -> np.ndarray:
    """
    Generate a noise map using Perlin noise.

    Parameters
    ----------
    size : int, optional
        The size of the noise map, which will be a square matrix of dimensions (size x size). Defaults to 100.
    seed : int, optional
        The seed for the noise generation, used to initialize the random number generator. Defaults to 0.
    octaves : int, optional
        The number of levels of detail in the noise. Higher values result in more detailed noise. Defaults to 3.
    persistence : float, optional
        The amplitude of each octave in the noise. This affects the roughness of the noise. Defaults to 0.75.
    lacunarity : float, optional
        The frequency of each octave in the noise. This affects the detail frequency. Defaults to 2.0.

    Returns
    -------
    np.ndarray
        A normalized noise matrix of shape (size, size), with values ranging from 0 to 1.
    """
    noise = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            noise[i][j] = pnoise2(
                i/size, j/size,
                octaves = octaves,
                persistence = persistence,
                lacunarity = lacunarity,
                base = seed
            )

    return Normalize_Noise(noise)

def Apply_Cutoff(noise: np.ndarray, cutoff: float) -> np.ndarray:
    """
    Apply a cutoff threshold to a noise matrix.

    Parameters
    ----------
    noise : np.ndarray
        The noise matrix to apply the cutoff to.
    cutoff : float
        The cutoff value. Elements of the noise matrix below this value will be set to 0.0.

    Returns
    -------
    np.ndarray
        The noise matrix with values below the cutoff set to 0.0.
    """
    return np.where(noise < cutoff, 0.0, noise)

def Convert_to_Coordinates(noise: np.ndarray) -> list[tuple[int, int]]:
    """
    Converts a noise matrix to a list of coordinate pairs.

    Parameters
    ----------
    noise : np.ndarray
        The noise matrix to convert.

    Returns
    -------
    list[tuple[int, int]]
        A list of coordinate pairs.
    """
    coordinates: list[tuple[int, int]] = []

    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            if noise[i][j] != 0:
                coordinates.append((i, j))

    return coordinates

def Adjust_Axis_Ranges(coordinates: list[tuple[int, int]], longitudeRange: tuple[float, float], latitudeRange: tuple[float, float], xLength: int, yLength: int) -> list[tuple[int, int]]:
    """
    Adjusts a list of coordinates to fit within a specific longitude and latitude range.

    Parameters
    ----------
    coordinates : list[tuple[int, int]]
        The list of coordinates to adjust.
    longitudeRange : tuple[float, float]
        The range of the longitude axis.
    latitudeRange : tuple[float, float]
        The range of the latitude axis.
    xLength : int
        The length of the x-axis.
    yLength : int
        The length of the y-axis.

    Returns
    -------
    list[tuple[int, int]]
        The adjusted list of coordinates.
    """
    adjustedCoordinates: list[tuple[int, int]] = []

    for coordinate in coordinates:
        adjustedCoordinates.append((
            Map_To_Range(longitudeRange[0], longitudeRange[1], coordinate[0], 0, xLength),
            Map_To_Range(latitudeRange[0], latitudeRange[1], coordinate[1], 0, yLength),
        ))

    return adjustedCoordinates