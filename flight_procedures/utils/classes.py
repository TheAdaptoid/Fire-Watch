from dataclasses import dataclass


@dataclass
class HotSpot:
    """
    A class representing a hot spot.
    """

    lattitude: float
    longitude: float


@dataclass
class HotZone:
    """
    A class representing a collection of hot spots.
    """

    id: int
    points: list[HotSpot]


@dataclass
class FlightPoint:
    """
    A class representing a flight point.
    """

    lattitude: float
    longitude: float
    altitude: float


@dataclass
class FlightPath:
    """
    A class representing a flight path.
    """

    points: list[FlightPoint]

    @property
    def first_point(self) -> FlightPoint:
        """
        The first point in the flight path.

        Returns:
            FlightPoint: The first point in the flight path.
        """
        return self.points[0]

    @property
    def last_point(self) -> FlightPoint:
        """
        The last point in the flight path.

        Returns:
            FlightPoint: The last point in the flight path.
        """
        return self.points[-1]
