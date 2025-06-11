use numpy::ndarray::parallel::prelude::*;
use numpy::ndarray::{s, Array, Array3, Axis};
use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::ops::Add;

#[pyfunction]
pub fn low_frame_rate<'py>(
    py: Python<'py>,
    frames: PyReadonlyArray3<f64>,
    m: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let frames = frames.as_array();
    let batch_size = frames.shape()[0];
    let n_frames = frames.shape()[1];
    let n_hidden = frames.shape()[2];
    let n_output_frames = n_frames.div_ceil(n);
    let n_output_hidden = n_hidden * m;
    let mut output_frames = Array3::<f64>::zeros((batch_size, n_output_frames, n_output_hidden));
    // repeat time = (m - 1) // 2
    let repeat_times = (m - 1) / 2;
    // 第一帧填充: 数组第一个元素 * repeat_times + 1 -> frames[1: repeat_times + 1]
    for i in 0..repeat_times + 1 {
        output_frames
            .slice_mut(s![.., 0, i * n_hidden..(i + 1) * n_hidden])
            .assign(&frames.slice(s![.., 0, ..]));
    }
    for i in 0..(m - repeat_times - 1) {
        let start = (repeat_times + i + 1) * n_hidden;
        let end = (repeat_times + i + 2) * n_hidden;
        output_frames
            .slice_mut(s![.., 0, start..end])
            .assign(&frames.slice(s![.., 1 + i, ..]));
    }

    // 填充剩余帧
    for i in 1..n_output_frames {
        // 起点在0点左侧
        if repeat_times > i * n {
            if repeat_times > i * n + m {
                // 此时全部用frames[0]填充
                for j in 0..m {
                    let inner_start = j * n_hidden;
                    let inner_end = (j + 1) * n_hidden;
                    output_frames
                        .slice_mut(s![.., i, inner_start..inner_end])
                        .assign(&frames.slice(s![.., 0, ..]));
                }
            } else {
                let n_left = repeat_times - i * n;
                // m - repeat_times个frames[0]填充
                for j in 0..n_left {
                    let inner_start = j * n_hidden;
                    let inner_end = (j + 1) * n_hidden;
                    output_frames
                        .slice_mut(s![.., i, inner_start..inner_end])
                        .assign(&frames.slice(s![.., 0, ..]));
                }

                for j in 0..m - n_left {
                    let inner_start = (j + n_left) * n_hidden;
                    let inner_end = (j + n_left + 1) * n_hidden;
                    output_frames
                        .slice_mut(s![.., i, inner_start..inner_end])
                        .assign(&frames.slice(s![.., j, ..]));
                }
            }
        }
        // 起点在0点右侧
        else {
            let start = i * n - repeat_times;
            for j in 0..m {
                let inner_start = j * n_hidden;
                let inner_end = (j + 1) * n_hidden;
                if start + j >= n_frames {
                    output_frames
                        .slice_mut(s![.., i, inner_start..inner_end])
                        .assign(&frames.slice(s![.., -1, ..]));
                } else {
                    output_frames
                        .slice_mut(s![.., i, inner_start..inner_end])
                        .assign(&frames.slice(s![.., start + j, ..]));
                }
            }
        }
    }
    Ok(PyArray3::from_owned_array(py, output_frames))
}

#[pyfunction]
pub fn compute_decibel<'py>(
    python: Python<'py>,
    frames: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let frame_len = frames.shape()[0];
    let mut buffer = Vec::<f64>::with_capacity(frame_len);
    frames
        .as_array()
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| row.pow2().sum().add(1e-6).log10() * 10.)
        .collect_into_vec(&mut buffer);
    let decibels = Array::<f64, _>::from(buffer)
        .into_shape_with_order((frame_len, 1))
        .unwrap();
    Ok(PyArray2::from_owned_array(python, decibels))
}

pub fn register_module(core_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let audio_module = PyModule::new(core_module.py(), "audio")?;
    audio_module.add_function(wrap_pyfunction!(compute_decibel, &audio_module)?)?;
    audio_module.add_function(wrap_pyfunction!(low_frame_rate, &audio_module)?)?;
    core_module.add_submodule(&audio_module)?;
    Ok(())
}
