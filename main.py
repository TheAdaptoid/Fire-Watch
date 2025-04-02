from krpc.client import Client
from flight_procedures.utils import HotSpot, HotZone, FlightPath, FlightPlan
from flight_procedures.processing import (
    generate_hot_spots,
    identify_hot_zones,
    create_flight_paths,
    simple_flight_plan,
    generate_waypoints,
    get_client,
    execute_flight_plan
)

CONNECTION: Client = get_client()
LEVEL_OF_DETAIL: int = 25
LATITDUE_RANGE: tuple[float, float] = (-82.5, -72.5)
LONGITUDE_RANGE: tuple[float, float] = (0, 7.5)


def main() -> None:
    """
    The main flight procedure entry point
    """

    # Create hot spots
    print("Generating hot spots...")
    hot_spots: list[HotSpot] = generate_hot_spots(
        latitude_range=LATITDUE_RANGE,
        longitude_range=LONGITUDE_RANGE,
        size=1000,
    )

    # Create clusters
    print("Identifying hot zones...")
    hot_zones: list[HotZone] = identify_hot_zones(hot_spots=hot_spots)

    # Create flight paths
    print("Creating flight paths...")
    flight_paths: list[FlightPath] = create_flight_paths(
        hot_zones=hot_zones, level_of_detail=LEVEL_OF_DETAIL
    )

    # Create flight plan
    print("Planning flight route...")
    flight_plan: FlightPlan = simple_flight_plan(flight_paths=flight_paths)

    # Create waypoints
    print("Generating waypoints...")
    generate_waypoints(client=CONNECTION, flight_plan=flight_plan)

    # Execute flight plan
    print("Executing flight plan...")
    execute_flight_plan(client=CONNECTION, flight_plan=flight_plan)



if __name__ == "__main__":
    main()
