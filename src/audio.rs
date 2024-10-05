use ndarray::prelude::*;
use ndarray::Array2;
use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use uuid::Uuid;

#[derive(Debug, Clone)]
#[pyclass(subclass)]
pub struct Audio {
    #[pyo3(get)]
    id: String,
    #[pyo3(get, set)]
    path: String,
    desc: Option<String>,
    duration: Option<f32>,
    mono: Option<bool>,
    sample_rate: Option<u16>,
    waveform: Option<Array2<f32>>,
}

#[pymethods]
impl Audio {
    #[new]
    fn new(path: String, desc: Option<String>) -> Self {
        let id = Uuid::new_v4().to_string();
        Self {
            id,
            path,
            desc,
            duration: None,
            mono: None,
            sample_rate: None,
            waveform: None,
        }
    }

    #[getter]
    fn duration(&self) -> Option<f32> {
        self.duration
    }

    #[setter]
    fn set_duration(&mut self, duration: Option<f32>) {
        self.duration = duration;
    }

    #[getter]
    fn mono(&self) -> Option<bool> {
        self.mono
    }

    #[setter]
    fn set_mono(&mut self, mono: Option<bool>) {
        self.mono = mono;
    }

    #[getter]
    fn sample_rate(&self) -> Option<u16> {
        self.sample_rate
    }

    #[setter]
    fn set_sample_rate(&mut self, sample_rate: Option<u16>) {
        self.sample_rate = sample_rate;
    }

    #[getter]
    fn desc(&self) -> Option<&str> {
        self.desc.as_deref()
    }

    #[setter]
    fn set_desc(&mut self, desc: Option<String>) {
        self.desc = desc;
    }

    #[getter]
    fn waveform<'py>(&mut self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        match &mut self.waveform {
            Some(waveform) => Some(waveform.to_pyarray_bound(py)),
            None => None,
        }
    }

    #[setter]
    fn set_waveform(&mut self, waveform: PyReadonlyArray2<f32>) {
        self.waveform = Some(waveform.as_array().to_owned());
    }
}

#[pyfunction]
pub fn low_frame_rate<'py>(
    py: Python<'py>,
    frames: PyReadonlyArray3<f32>,
    m: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let frames = frames.as_array();
    let batch_size = frames.shape()[0];
    let n_frames = frames.shape()[1];
    let n_hidden = frames.shape()[2];
    let n_output_frames = n_frames.div_ceil(n);
    let n_output_hidden = n_hidden * m;
    let mut output_frames = Array3::<f32>::zeros((batch_size, n_output_frames, n_output_hidden));
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
    Ok(PyArray3::from_owned_array_bound(py, output_frames))
}
