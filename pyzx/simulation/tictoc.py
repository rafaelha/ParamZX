import time

global_time_point = [0]


def tic():
    now = time.perf_counter()
    global_time_point[0] = now


def toc(message: str = "Elapsed"):
    now = time.perf_counter()
    duration = now - global_time_point[0]
    global_time_point[0] = now
    print(f"{message}: {duration:.2f}s")
    return duration
