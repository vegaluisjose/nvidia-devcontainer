import torch

# Check if CUDA is available
print(f"CUDA is available: {torch.cuda.is_available()}") # True if CUDA is available, False otherwise

# Get the current CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

# Get the number of available CUDA devices
num_devices = torch.cuda.device_count()
print(f"Number of CUDA devices: {num_devices}")

# Get the name of the current CUDA device
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    print(f"Name of the current CUDA device: {torch.cuda.get_device_name(current_device)}")

# Create a tensor on the GPU
if torch.cuda.is_available():
    tensor = torch.randn(3, 3).to(device)
    print(f"Tensor on {device}: {tensor}")
