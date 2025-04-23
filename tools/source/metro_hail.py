def main(start_point: str, end_point: str) -> float:
    """
    Calculate the fare for a trip between two points.

    :param start_point: str: The starting location for the ride.
    :param end_point: str: The destination location for the ride.
    :returns: float: The estimated fare for the journey.
    """
    base_fare = 14.0
    distance = len(start_point) + len(end_point)
    fare = base_fare + distance
    return fare
