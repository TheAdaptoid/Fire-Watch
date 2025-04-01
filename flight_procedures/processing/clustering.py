from sklearn.cluster import DBSCAN
from numpy import array, float64
from numpy.typing import NDArray

from flight_procedures.utils import HotSpot, HotZone

EPS: float = 0.015
MIN_SAMPLES: int = 25


def fit_clustering_model(
    hot_spots: list[HotSpot], eps: float, min_samples: int
) -> DBSCAN:
    """
    Fit hot spot coordinates to a clustering model.

    Args:
        hot_spots (list[HotSpot]): A list of HotSpot objects.
        eps (float): The epsilon value for the clustering model.
        min_samples (int): The minimum samples value for the clustering model.

    Returns:
        DBSCAN: The fitted clustering model.
    """

    # Prepare model
    model: DBSCAN = DBSCAN(eps=eps, min_samples=min_samples)

    # convert hot spot list to matrix-like
    matrix: list[list[float]] = [
        [hot_spot.lattitude, hot_spot.longitude] for hot_spot in hot_spots
    ]
    matrix_like: NDArray[float64] = array(object=matrix)

    # fit model
    model.fit(matrix_like)

    return model


def extract_hot_zones(
    hot_spots: list[HotSpot], clustering_model: DBSCAN
) -> list[HotZone]:
    """
    Creates a list of HotZone objects.

    Args:
        hot_spots (list[HotSpot]): A list of HotSpot objects.
        clustering_model (DBSCAN): The clustering model.

    Returns:
        list[HotZone]: A list of HotZone objects.
    """

    zone_ids: list[int] = array(clustering_model.labels_).tolist()
    zone_dict: dict[int, list[HotSpot]] = {zone_id: [] for zone_id in zone_ids}

    # build zone dictionary
    for hot_spot_index, zone_id in enumerate(zone_ids):
        zone_dict[zone_id].append(hot_spots[hot_spot_index])

    # build hot zones
    hot_zones: list[HotZone] = [
        HotZone(id=zone_id, points=zone) for zone_id, zone in zone_dict.items()
    ]

    return hot_zones


def identify_hot_zones(
    hot_spots: list[HotSpot], eps: float = EPS, min_samples: int = MIN_SAMPLES
) -> list[HotZone]:
    """
    Creates a list of HotZone objects.

    Args:
        hot_spots (list[HotSpot]): A list of HotSpot objects.
        eps (float, optional): The maximum distance between two samples for
            them to be considered as in the same
            neighborhood. Defaults to EPS.
        min_samples (int, optional): The number of samples in a neighborhood
            for a point to be considered as a core point.
            Defaults to MIN_SAMPLES.

    Returns:
        list[HotZone]: A list of HotZone objects.
    """

    model: DBSCAN = fit_clustering_model(hot_spots, eps, min_samples)
    hot_zones: list[HotZone] = extract_hot_zones(hot_spots, model)

    return hot_zones
