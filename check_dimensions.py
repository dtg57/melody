from functools import wraps


def CheckDimensions(dimensions: list[float]):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            output = function(*args, **kwargs)
            outputShape = list(output.shape)
            if outputShape != dimensions:
                raise RuntimeError("Tensor dimensions incorrect, expected " + str(dimensions) +  " got " + str(outputShape))
            return output
        return wrapper
    return decorator
