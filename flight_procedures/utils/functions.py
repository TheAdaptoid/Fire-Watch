from time import time

def timed_function(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute.\n")
        return result

    return wrapper