import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from numpy import array, ndarray
from flight_procedures.utils import HotZone, FlightPath, FlightPoint, FlightPlan
from flight_procedures.utils import timed_function


def fit_linear_model(
    x_coordinates: list[float], y_coordinates: list[float], level_of_detail: int
) -> tuple[ndarray, ndarray]:
    """
    Fits a polynomial regression model to the given x and y coordinates and generates new points
    based on the specified level of detail.

    Args:
        x_coordinates (list[float]): A list of x coordinates.
        y_coordinates (list[float]): A list of y coordinates corresponding to x coordinates.
        level_of_detail (int): The number of points to generate for the resultant flight path.

    Returns:
        tuple[ndarray, ndarray]: A tuple of ndarrays containing the generated x and y coordinates.
    """

    # Create polynomial features
    feature_builder: PolynomialFeatures = PolynomialFeatures(degree=5)
    x_features: ndarray = feature_builder.fit_transform(
        array(x_coordinates).reshape(-1, 1)
    )

    # Fit model
    linear_regression: LinearRegression = LinearRegression()
    linear_regression.fit(x_features, y_coordinates)

    # Generate n points
    n_x_coordinates: ndarray = np.linspace(
        min(x_coordinates), max(x_coordinates), level_of_detail
    )
    n_y_coordinates: ndarray = linear_regression.predict(
        feature_builder.fit_transform(n_x_coordinates.reshape(-1, 1))
    )

    return n_x_coordinates, n_y_coordinates


def calculate_flight_path(hot_zone: HotZone, level_of_detail: int) -> FlightPath:
    """
    Calculates a flight path for a given hot zone.

    Args:
        hot_zone (HotZone): The hot zone to generate a flight path for.
        level_of_detail (int): The number of points to generate for the resultant flight path.

    Returns:
        FlightPath: A flight path containing the generated points.
    """

    # Convert points to matrix-like
    x_coordinates: list[float] = [hot_spot.lattitude for hot_spot in hot_zone.points]
    y_coordinates: list[float] = [hot_spot.longitude for hot_spot in hot_zone.points]

    # Fit linear model
    n_x_coordinates, n_y_coordinates = fit_linear_model(
        x_coordinates, y_coordinates, level_of_detail
    )

    # Create flight path
    flight_points: list[FlightPoint] = [
        FlightPoint(latitude=point[0], longitude=point[1], altitude=0)
        for point in zip(n_x_coordinates, n_y_coordinates)
    ]
    flight_path: FlightPath = FlightPath(points=flight_points)

    return flight_path


@timed_function
def create_flight_paths(
    hot_zones: list[HotZone], level_of_detail: int = 100
) -> list[FlightPath]:
    """
    Generates flight paths for a list of hot zones.

    Args:
        hot_zones (list[HotZone]): A list of HotZone objects for which to generate flight paths.
        level_of_detail (int, optional): The number of points to generate for each flight path.
            Defaults to 100.

    Returns:
        list[FlightPath]: A list of FlightPath objects corresponding to the provided hot zones.
    """

    flight_paths: list[FlightPath] = [
        calculate_flight_path(hot_zone, level_of_detail) for hot_zone in hot_zones
    ]

    return flight_paths


@timed_function
def simple_flight_plan(flight_paths: list[FlightPath]) -> FlightPlan:
    """
    Creates a simple flight plan from a list of flight paths.

    Args:
        flight_paths (list[FlightPath]): A list of FlightPath objects to be included in the flight plan.

    Returns:
        FlightPlan: A FlightPlan object containing the generated flight points.
    """
    route_set: list[FlightPoint] = []
    for flight_path in flight_paths:
        route_set.extend(flight_path.points)

    return FlightPlan(route_set=route_set)
