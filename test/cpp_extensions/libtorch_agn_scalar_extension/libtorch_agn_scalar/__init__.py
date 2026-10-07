from pathlib import Path

import torch


for library in Path(__file__).parent.glob("_C.*"):
    if library.suffix in (".so", ".pyd", ".dylib"):
        torch.ops.load_library(str(library))
