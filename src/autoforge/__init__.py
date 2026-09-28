import os

# On Apple Silicon, run the odd op that has no Metal kernel yet on the CPU
# instead of failing the whole run. Must be set before torch is imported;
# no effect on the other backends.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
