def annealing(t, warm_up_interval = 10000, anneal = False):
    if not anneal:
        return 1.0
    else:
        return min(1.0, 0.1 + t/warm_up_interval)
    