use asr_data::{AudioSource, Waveform};
use numpy::ndarray::parallel::prelude::*;
use numpy::ndarray::{s, Array, Array1, Array2, Array3, Axis};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::ops::Add;

/// Load and transform a waveform through asr-data's Rust audio pipeline.
///
/// Args:
///     uri: Source path or URL.
///     target_sample_rate: Optional output sample rate.
///     mono: Whether to downmix the waveform to one channel.
///
/// Returns:
///     A decoded and optionally transformed waveform.
///
/// Examples:
///     ```ignore
///     let waveform = load_rust_waveform("example.wav".to_string(), Some(16_000), Some(true))?;
///     assert_eq!(waveform.sample_rate, 16_000);
///     ```
fn load_rust_waveform(
    uri: String,
    target_sample_rate: Option<u32>,
    mono: Option<bool>,
) -> anyhow::Result<Waveform> {
    if target_sample_rate == Some(0) {
        anyhow::bail!("sample_rate must be greater than zero");
    }

    let source = AudioSource::new(uri);
    let mut waveform = Waveform::from_source(&source)?;
    if mono == Some(true) && waveform.channels != 1 {
        waveform = waveform.to_mono()?;
    }
    if let Some(sample_rate) = target_sample_rate {
        if waveform.sample_rate != sample_rate {
            waveform = waveform.resample(sample_rate)?;
        }
    }
    Ok(waveform)
}

/// Convert an interleaved asr-data waveform into osc-data's NumPy layout.
///
/// Args:
///     py: Active Python interpreter token.
///     waveform: Decoded waveform with interleaved samples.
///
/// Returns:
///     A one-dimensional mono or two-dimensional multi-channel NumPy array.
///
/// Examples:
///     ```ignore
///     let array = waveform_to_numpy(py, waveform)?;
///     assert!(array.hasattr("shape")?);
///     ```
fn waveform_to_numpy<'py>(py: Python<'py>, waveform: Waveform) -> PyResult<Bound<'py, PyAny>> {
    let channels = usize::from(waveform.channels);
    if channels == 0 {
        return Err(PyValueError::new_err(
            "audio channel count must be greater than zero",
        ));
    }
    if !waveform.samples.len().is_multiple_of(channels) {
        return Err(PyValueError::new_err(
            "audio samples do not contain complete channel frames",
        ));
    }

    if channels == 1 {
        return Ok(PyArray1::from_owned_array(py, Array1::from_vec(waveform.samples)).into_any());
    }

    // asr-data stores interleaved samples for efficient decoding and streaming.
    // Keep osc-data's existing public NumPy layout: (channels, samples).
    let frame_count = waveform.samples.len() / channels;
    let mut planar = vec![0.0_f32; waveform.samples.len()];
    for (frame_index, frame) in waveform.samples.chunks_exact(channels).enumerate() {
        for (channel_index, sample) in frame.iter().enumerate() {
            planar[channel_index * frame_count + frame_index] = *sample;
        }
    }
    let array = Array2::from_shape_vec((channels, frame_count), planar)
        .map_err(|error| PyValueError::new_err(format!("failed to create audio array: {error}")))?;
    Ok(PyArray2::from_owned_array(py, array).into_any())
}

/// Load an audio source and optionally convert its sample rate and channel count.
///
/// Args:
///     uri: Local path, file URL, HTTP(S) URL, or another source supported by asr-data.
///     sample_rate: Optional target sample rate. The native rate is preserved when omitted.
///     mono: If True, downmix all channels to mono. Other values preserve the channel layout.
///
/// Returns:
///     A tuple of `(numpy_array, sample_rate)`. Mono audio uses shape `(samples,)` and
///     multi-channel audio uses shape `(channels, samples)`.
///
/// Raises:
///     ValueError: If the source cannot be loaded, the target rate is invalid, or the
///         decoded samples are malformed.
///
/// Examples:
///     >>> data, sample_rate = load_audio("example.wav", sample_rate=16000, mono=True)
///     >>> data.ndim
///     1
#[pyfunction]
#[pyo3(signature = (uri, sample_rate=None, mono=None))]
pub fn load_audio<'py>(
    py: Python<'py>,
    uri: String,
    sample_rate: Option<u32>,
    mono: Option<bool>,
) -> PyResult<(Bound<'py, PyAny>, u32)> {
    let waveform = py
        .allow_threads(move || load_rust_waveform(uri, sample_rate, mono))
        .map_err(|error| PyValueError::new_err(format!("failed to load audio: {error:#}")))?;
    let actual_sample_rate = waveform.sample_rate;
    let data = waveform_to_numpy(py, waveform)?;
    Ok((data, actual_sample_rate))
}

/// Low frame rate processing for audio frames.
///
/// This function processes audio frames by repeating and concatenating frames
/// to achieve a lower effective frame rate.
///
/// Args:
///     frames: Input array of shape (batch_size, n_frames, n_hidden)
///     m: Number of times to repeat each frame segment
///     n: Downsampling factor for frames
///
/// Returns:
///     Processed array of shape (batch_size, n_output_frames, n_output_hidden)
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

/// Compute decibel values from audio frames.
///
/// Calculates the power in decibels for each audio frame using the formula:
///     dB = 10 * log10(sum(frame^2) + 1e-6)
///
/// Args:
///     frames: Input array of audio frames with shape (frame_len, features)
///
/// Returns:
///     Decibel values with shape (frame_len, 1)
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
    audio_module.add_function(wrap_pyfunction!(load_audio, &audio_module)?)?;
    audio_module.add_function(wrap_pyfunction!(compute_decibel, &audio_module)?)?;
    audio_module.add_function(wrap_pyfunction!(low_frame_rate, &audio_module)?)?;
    core_module.add_submodule(&audio_module)?;
    Ok(())
}
