
import torch
import time
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
# Using a slightly smaller size for faster initialization/teardown
a = torch.randn((6000, 6000), device=device)
b = torch.randn((6000, 6000), device=device)

while True:
    _ = torch.matmul(a, b)
    time.sleep(0.01)
