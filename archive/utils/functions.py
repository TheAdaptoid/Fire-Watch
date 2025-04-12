from time import time


def timed_function(func):
    """Decorator to time the execution of a function."""

    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(
            f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute.\n"
        )
        return result

    return wrapper
