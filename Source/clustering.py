from sklearn.cluster import DBSCAN
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from numpy import array, float64, ndarray
from numpy.typing import NDArray
import numpy as np

EPS: float = 0.05
MIN_SAMPLES: int = 5


class Zone:
    """
    A class representing a collection of hot spots.
    """

    def __init__(self, points: list[tuple[float, float]]):
        # Sort points by x
        # points.sort(key=lambda point: point[0])
        self.__points = points

    def build_flight_path(
        self, max_degree: int = 5, n_points: int = 100
    ) -> tuple[list[float], list[float]]:
        """
        Generates a set of `n` points that approximate the optimal flight path of the zone.

        Args:
            max_degree (int, optional): The maximum degree of the polynomials used during regression. Defaults to 5.
            n_points (int, optional): The number of points to generate. Defaults to 100.

        Returns:
            tuple[list[float], list[float]]: A tuple containing the x and y coordinates of the generated points.
        """
        # convert points to matrix-like
        x_coordinates: list[float] = [point[0] for point in self.__points]
        y_coordinates: list[float] = [point[1] for point in self.__points]

        # create polynomial features
        feature_builder: PolynomialFeatures = PolynomialFeatures(degree=max_degree)
        x_features: NDArray[float64] = feature_builder.fit_transform(
            array(x_coordinates).reshape(-1, 1)
        )

        # fit model
        linear_regression: LinearRegression = LinearRegression()
        linear_regression.fit(x_features, y_coordinates)

        # generate n points
        n_x_coordinates: NDArray = np.linspace(
            min(x_coordinates), max(x_coordinates), n_points
        )
        n_x_features: ndarray = feature_builder.transform(
            n_x_coordinates.reshape(-1, 1)
        )  # TODO: fix typing issue

        # predict y coordinates
        n_y_coordinates: NDArray[float64] = linear_regression.predict(n_x_features)

        return list(n_x_coordinates), list(n_y_coordinates)

    def get_points(self) -> list[tuple[float, float]]:
        """
        Returns the points of the zone.

        Returns:
            list[tuple[float, float]]: A list of tuples containing the x and y coordinates of the points.
        """
        return self.__points


class Zone_Builder:
    """
    A utility class for creating zones from a set of grid points.
    """

    def __init__(self, eps: float = EPS, min_samples: int = MIN_SAMPLES):
        # Private Attributes
        self.__grid_points: list[tuple[float, float]] = []
        self.__zones: list[Zone] = []
        self.__clustering_model: DBSCAN = DBSCAN(eps=eps, min_samples=min_samples)

    def define_grid(self, grid_points: list[tuple[float, float]]) -> None:
        """
        Sets the grid points for the zone builder.

        Args:
            grid_points (list[tuple[float, float]]): The grid points to set.
        """
        self.__grid_points = grid_points

    def __fit_coordinates(self) -> None:
        """
        Fits the coordinates to the clustering model.
        """
        matrix_like_coordinates: NDArray[float64] = array(object=self.__grid_points)

        # Fit model
        self.__clustering_model.fit(X=matrix_like_coordinates)

    def build_zones(self) -> None:
        """
        Creates zone objects for each cluster of hot spots identified.
        """
        # Initialize cluster dictionary
        cluster_dict: dict[int, list[tuple[float, float]]] = {}

        # Fit coordinates
        self.__fit_coordinates()

        # Extract cluster labels
        cluster_labels: list[int] = array(
            self.__clustering_model.labels_
        ).tolist()  # TODO: fix typing issue

        # Build cluster dictionary
        for coordinate_index, label in enumerate(cluster_labels):
            if label in cluster_dict:
                cluster_dict[label].append(self.__grid_points[coordinate_index])
            else:
                cluster_dict[label] = [self.__grid_points[coordinate_index]]

        # Remove empty clusters
        del cluster_dict[-1]

        # Build zones
        self.__zones = [Zone(points=cluster) for cluster in cluster_dict.values()]

    def get_zones(self) -> list[Zone]:
        """
        Returns the zones created by the zone builder.

        Returns:
            list[Zone]: A list of Zone objects.
        """
        return self.__zones
