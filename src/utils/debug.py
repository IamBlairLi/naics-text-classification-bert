def peek(name, x, n=5):
    import numpy as np
    print(f"\n--- {name} ---")
    print("type:", type(x))
    if hasattr(x, "shape"):
        print("shape:", x.shape)
    if hasattr(x, "dtype"):
        print("dtype:", x.dtype)
    # tensor -> cpu numpy
    try:
        import torch
        if isinstance(x, torch.Tensor):
            y = x.detach().cpu().numpy()
        else:
            y = np.array(x)
        flat = y.flatten()
        print("first:", flat[:n])
        print("min/max:", float(flat.min()), float(flat.max()))
    except Exception:
        print("preview:", str(x)[:200])
