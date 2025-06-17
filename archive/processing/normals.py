MACH_SPEED: float = 340.3

def normalize(value: float, min_value: float, max_value: float) -> float:
    numerator: float = value - min_value
    denominator: float = max_value - min_value

    if denominator == 0:
        denominator = 0.001

    return numerator / denominator

def square(x: float) -> float:
    return x * x

def magnitude(x: float, y: float, z: float) -> float:
    return (square(x) + square(y) + square(z)) ** 0.5