import numpy as np
import torch
import time


def torch_lfr(x: np.ndarray, lfr_m: int, lfr_n: int):
    x = torch.from_numpy(x)
    start = time.perf_counter()
    LFR_inputs = []
    T = x.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = x[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, x))
    T = T + (lfr_m - 1) // 2
    for i in range(T_lfr):
        if lfr_m <= T - i * lfr_n:
            LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).view(1, -1))
        else:  # process last LFR frame
            num_padding = lfr_m - (T - i * lfr_n)
            frame = (inputs[i * lfr_n :]).view(-1)
            for _ in range(num_padding):
                frame = torch.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    LFR_outputs = torch.vstack(LFR_inputs)
    outputs = LFR_outputs.type(torch.float32)
    end = time.perf_counter()
    spent = end - start
    return outputs.numpy(), spent


def numpy_lfr(x: np.ndarray, lfr_m: int, lfr_n: int):
    start = time.perf_counter()
    LFR_inputs = []
    T = x.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = np.tile(x[0], ((lfr_m - 1) // 2, 1))
    inputs = np.vstack((left_padding, x))
    T = T + (lfr_m - 1) // 2
    for i in range(T_lfr):
        if lfr_m <= T - i * lfr_n:
            LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
        else:
            # process last LFR frame
            num_padding = lfr_m - (T - i * lfr_n)
            frame = inputs[i * lfr_n :].reshape(-1)
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))

            LFR_inputs.append(frame)
    LFR_outputs = np.vstack(LFR_inputs).astype(np.float32)
    end = time.perf_counter()
    spent = end - start
    return LFR_outputs, spent


def lfr(x: np.ndarray, lfr_m: int, lfr_n: int):
    from osc_data.audio import low_frame_rate

    start = time.perf_counter()
    y = low_frame_rate(x, lfr_m, lfr_n)
    end = time.perf_counter()
    spent = end - start
    return y, spent


if __name__ == "__main__":
    import numpy as np

    arr = np.random.randn(2, 1000, 10000)
    lfr_m = 5
    lfr_n = 1
    r1 = []
    r1_spent = []
    r2 = []
    r2_spent = []
    for i in range(2):
        rr1, rr1_spent = torch_lfr(arr[i], lfr_m, lfr_n)
        r1.append(rr1)
        r1_spent.append(rr1_spent)
        rr2, rr2_spent = numpy_lfr(arr[i], lfr_m, lfr_n)
        r2.append(rr2)
        r2_spent.append(rr2_spent)
    r1 = np.array(r1)
    r1_spent = sum(r1_spent)
    r2 = np.array(r2)
    r2_spent = sum(r2_spent)
    r3, r3_spent = lfr(arr, lfr_m, lfr_n)
    assert np.allclose(r1, r3), "not equal"
    assert np.allclose(r2, r3), "not equal"
    print(f"torch_lfr: {r1_spent}, numpy_lfr: {r2_spent}, rust_lfr: {r3_spent}")
    print(f"对比torch: {r1_spent / r3_spent}")
    print(f"对比numpy: {r2_spent / r3_spent}")
