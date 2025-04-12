from dataclasses import dataclass

@dataclass
class Coordinate:
    """
    A class representing a coordinate.
    """

    latitude: float
    longitude: float

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

    latitude: float
    longitude: float
    altitude: float

    def __str__(self):
        return f"({self.latitude}, {self.longitude}, {self.altitude})"
    
    def __repr__(self):
        return self.__str__()


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


@dataclass
class FlightPlan:
    """
    A class representing a flight plan.
    """

    route_set: list[FlightPoint]


@dataclass
class FlightDataPacket:
    """
    A class representing a packet of flight data.
    """

    # Altitude
    altitude_asl: float  # meters
    altitude_agl: float  # meters

    # Location
    latitude: float
    longitude: float

    # Orientation
    pitch: float  # -90 to 90, degrees
    heading: float  # 0 to 360, degrees
    roll: float  # -180 to 180, degrees

    # Velocity
    velocity_x: float
    velocity_y: float
    velocity_z: float
    velocity_magnitude: float  # meters per second
    horizontal_speed: float
    vertical_speed: float

    # Other
    g_force: float
    aoa: float

    def to_vector(self) -> list[float]:
        """
        Converts the flight data to a vector of floats.

        Returns:
            list[float]: A list of floats containing the flight data.
        """
        return [
            self.altitude_asl,
            self.altitude_agl,
            self.latitude,
            self.longitude,
            self.pitch,
            self.heading,
            self.roll,
            self.velocity_x,
            self.velocity_y,
            self.velocity_z,
            self.velocity_magnitude,
            self.horizontal_speed,
            self.vertical_speed,
            self.g_force,
            self.aoa,
        ]
