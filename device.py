import torch

# Cihazı platforma göre otomatik olarak belirle
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Cihaz olarak Apple MPS (GPU) seçildi.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cihaz olarak NVIDIA CUDA (GPU) seçildi.")
else:
    device = torch.device("cpu")
    print("Cihaz olarak CPU seçildi.")
