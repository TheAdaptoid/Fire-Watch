import krpc
from krpc.client import Client
from flight_procedures.utils import HotSpot, HotZone, FlightPath
from flight_procedures.processing import generate_hot_spots, identify_hot_zones, create_flight_paths

# CONNECTION: Client = krpc.connect(name="Flight Procedure")
LEVEL_OF_DETAIL: int = 25


def main() -> None:
    """
    The main flight procedure entry point
    """

    # Create hot spots
    print("Generating hot spots...")
    hot_spots: list[HotSpot] = generate_hot_spots(
        latitude_range=(-78, -73),
        longitude_range=(5, 11),
        size=1000,
    )

    # Create clusters
    print("Identifying hot zones...")
    hot_zones: list[HotZone] = identify_hot_zones(hot_spots=hot_spots)

    # Create waypoints

    # Create flight paths
    print("Creating flight paths...")
    flight_paths: list[FlightPath] = create_flight_paths(
        hot_zones=hot_zones,
        level_of_detail=LEVEL_OF_DETAIL
    )

    # Create flight plan

    # Execute flight plan


if __name__ == "__main__":
    main()
