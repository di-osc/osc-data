import time
import numpy as np


def torch_compute_decibel(x: np.ndarray):
    import torch

    x = torch.from_numpy(x)
    start = time.perf_counter()
    outputs = x.square().sum(dim=1, keepdim=True).add(1e-6).log10() * 10
    end = time.perf_counter()
    spent = end - start
    return outputs, spent


def numpy_compute_decibel(x):
    start = time.perf_counter()
    y = np.log10(np.sum(x**2, axis=1, keepdims=True) + 1e-6) * 10
    end = time.perf_counter()
    spent = end - start
    return y, spent


def compute_decibel(x: np.ndarray):
    from osc_data.audio import compute_decibel

    start = time.perf_counter()
    y = compute_decibel(x)
    end = time.perf_counter()
    spent = end - start
    return y, spent


if __name__ == "__main__":
    arr = np.random.rand(10000, 10000)
    r1, torch_spent = torch_compute_decibel(arr)
    r2, np_spent = numpy_compute_decibel(arr)
    r3, rs_spent = compute_decibel(arr)
    print(f"torch_spent: {torch_spent}, np_spent: {np_spent}, rs_spent: {rs_spent}")
    # 打印提升比例
    print(f"对比torch： {torch_spent / rs_spent}")
    print(f"对比numpy： {np_spent / rs_spent}")
    assert np.allclose(r1, r2), "Results are not close"
    assert np.allclose(r2, r3), "Results are not close"
