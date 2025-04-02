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

    latitude: float
    longitude: float
    altitude: float

    def __str__(self) -> str:
        return f"({self.latitude}, {self.longitude}, {self.altitude})"

    def __repr__(self) -> str:
        return f"FlightPoint(lattitude={self.latitude}, longitude={self.longitude}, altitude={self.altitude})"


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
class DataPacket:
    """
    A class representing a packet of flight data.
    """

    # Altitude
    altitude_asl: float # meters
    altitude_agl: float # meters

    # Location
    latitude: float
    longitude: float

    # Orientation
    pitch: float # -90 to 90, degrees
    heading: float # 0 to 360, degrees
    roll: float # -180 to 180, degrees

    # Velocity
    velocity_x: float
    velocity_y: float
    velocity_z: float
    velocity_magnitude: float # meters per second
    horizontal_speed: float
    vertical_speed: float

    # Other
    g_force: float
    aoa: float

    def to_vector(self) -> list[float]:
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
            self.aoa
        ]

@dataclass
class InputPacket(DataPacket):

    target_altitude: float
    target_lattitude: float
    target_longitude: float

    @classmethod
    def from_data_packet(cls, data_packet: DataPacket, target_point: FlightPoint) -> "InputPacket":
        return cls(
            altitude_asl=data_packet.altitude_asl,
            altitude_agl=data_packet.altitude_agl,
            latitude=data_packet.latitude,
            longitude=data_packet.longitude,
            pitch=data_packet.pitch,
            heading=data_packet.heading,
            roll=data_packet.roll,
            velocity_x=data_packet.velocity_x,
            velocity_y=data_packet.velocity_y,
            velocity_z=data_packet.velocity_z,
            velocity_magnitude=data_packet.velocity_magnitude,
            horizontal_speed=data_packet.horizontal_speed,
            vertical_speed=data_packet.vertical_speed,
            g_force=data_packet.g_force,
            aoa=data_packet.aoa,
            target_altitude=target_point.altitude,
            target_lattitude=target_point.latitude,
            target_longitude=target_point.longitude
        )
    
    def to_vector(self) -> list[float]:
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
            self.target_altitude,
            self.target_lattitude,
            self.target_longitude
        ]