import torch

def group_count(channels, maximum=32):
    for count in range(min(channels, maximum), 0, -1):
        if channels % count == 0:
            return count
    return 1
