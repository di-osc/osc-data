use anyhow::{anyhow, Context, Result};
use numpy::PyArray4;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::Deserialize;
use std::process::Command;
use video_rs::decode::Decoder;
use video_rs::Url;

#[derive(Debug, Deserialize)]
struct FfprobeStreams {
    streams: Vec<FfStream>,
    #[serde(default)]
    format: Option<FfFormat>,
}

#[derive(Debug, Deserialize)]
struct FfStream {
    #[serde(default)]
    width: Option<usize>,
    #[serde(default)]
    height: Option<usize>,
    #[serde(default)]
    avg_frame_rate: Option<String>,
    #[serde(default)]
    r_frame_rate: Option<String>,
    #[serde(default)]
    duration: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FfFormat {
    #[serde(default)]
    duration: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FfprobeFrames {
    #[serde(default)]
    frames: Vec<FfFrame>,
}

#[derive(Debug, Deserialize)]
struct FfFrame {
    #[serde(default)]
    key_frame: Option<i32>,
    #[serde(default)]
    pict_type: Option<String>,
    #[serde(default)]
    pkt_pts_time: Option<String>,
    #[serde(default)]
    best_effort_timestamp_time: Option<String>,
    #[serde(default)]
    pkt_size: Option<String>,
}

#[derive(Debug, Clone)]
struct VideoMeta {
    width: usize,
    height: usize,
    fps: f64,
    duration: f64,
}

fn parse_rate(rate: &str) -> Option<f64> {
    if rate.trim().is_empty() {
        return None;
    }
    let mut parts = rate.split('/');
    let num: f64 = parts.next()?.parse().ok()?;
    match parts.next() {
        Some(den) => {
            let den: f64 = den.parse().ok()?;
            if den == 0.0 {
                None
            } else {
                Some(num / den)
            }
        }
        None => Some(num),
    }
}

fn parse_f64_opt(s: &Option<String>) -> Option<f64> {
    s.as_ref()?.parse::<f64>().ok()
}

fn ffprobe_meta(input: &str) -> Result<VideoMeta> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration")
        .arg("-of")
        .arg("json")
        .arg(input)
        .output()
        .with_context(|| "failed to execute ffprobe")?;
    if !output.status.success() {
        return Err(anyhow!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let parsed: FfprobeStreams =
        serde_json::from_slice(&output.stdout).with_context(|| "failed to parse ffprobe json")?;
    let stream = parsed
        .streams
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no video stream found"))?;
    let width = stream.width.ok_or_else(|| anyhow!("missing width"))?;
    let height = stream.height.ok_or_else(|| anyhow!("missing height"))?;
    let fps = stream
        .avg_frame_rate
        .as_ref()
        .and_then(|r| parse_rate(r))
        .or_else(|| stream.r_frame_rate.as_ref().and_then(|r| parse_rate(r)))
        .ok_or_else(|| anyhow!("missing fps"))?;
    let duration = parse_f64_opt(&stream.duration)
        .or_else(|| parsed.format.and_then(|f| parse_f64_opt(&f.duration)))
        .unwrap_or(0.0);
    Ok(VideoMeta { width, height, fps, duration })
}

fn videors_decode_all(input: &str) -> Result<(Vec<u8>, usize, usize, usize, f64, f64)> {
    video_rs::init().map_err(|e| anyhow!(format!("video-rs init failed: {e:?}")))?;
    let url = if input.starts_with("http://") || input.starts_with("https://") || input.starts_with("rtsp://") {
        input.parse::<Url>().map_err(|e| anyhow!(format!("invalid url: {e}")))?
    } else {
        Url::from_file_path(input).map_err(|_| anyhow!("invalid file path"))?
    };

    let mut decoder = Decoder::new(url).map_err(|e| anyhow!(format!("decoder new failed: {e:?}")))?;
    let mut width: usize = 0;
    let mut height: usize = 0;
    let mut bytes: Vec<u8> = Vec::new();
    let mut frames: usize = 0;
    let mut first_ts: Option<f64> = None;
    let mut last_ts: Option<f64> = None;

    for res in decoder.decode_iter() {
        let (ts, frame) = match res {
            Ok(x) => x,
            Err(e) => return Err(anyhow!(format!("decode error: {e:?}"))),
        };
        let shape = frame.shape();
        if shape.len() != 3 { return Err(anyhow!("unexpected frame dims")); }
        let h = shape[0];
        let w = shape[1];
        let c = shape[2];
        if c != 3 { return Err(anyhow!("expected RGB channels=3")); }
        if width == 0 { width = w; }
        if height == 0 { height = h; }
        if w != width || h != height { return Err(anyhow!("variable frame size not supported")); }
        if let Some(slice) = frame.as_slice() {
            bytes.extend_from_slice(slice);
        } else {
            let owned = frame.to_owned();
            bytes.extend_from_slice(owned.as_slice().ok_or_else(|| anyhow!("failed to get owned slice"))?);
        }
        frames += 1;

        let ts_sec: f64 = ts.as_secs_f64();
        if first_ts.is_none() { first_ts = Some(ts_sec); }
        last_ts = Some(ts_sec);
    }

    if width == 0 || height == 0 { return Err(anyhow!("no frames decoded")); }
    let duration = match (first_ts, last_ts) { (Some(a), Some(b)) if b >= a => b - a, _ => 0.0 };
    let fps = if frames > 1 && duration > 0.0 { (frames as f64 - 1.0) / duration } else { 0.0 };
    Ok((bytes, frames, width, height, fps, duration))
}

// ffmpeg CLI path removed in favor of video-rs

fn videors_keyframes(input: &str) -> Result<Vec<(usize, f64, String, usize)>> {
    video_rs::init().map_err(|e| anyhow!(format!("video-rs init failed: {e:?}")))?;
    let url = if input.starts_with("http://") || input.starts_with("https://") || input.starts_with("rtsp://") {
        input.parse::<Url>().map_err(|e| anyhow!(format!("invalid url: {e}")))?
    } else {
        Url::from_file_path(input).map_err(|_| anyhow!("invalid file path"))?
    };

    let mut decoder = Decoder::new(url).map_err(|e| anyhow!(format!("decoder new failed: {e:?}")))?;
    let time_base = decoder.time_base();
    let mut out: Vec<(usize, f64, String, usize)> = Vec::new();
    let mut index: usize = 0;
    for res in decoder.decode_raw_iter() {
        let raw = match res { Ok(f) => f, Err(e) => return Err(anyhow!(format!("decode error: {e:?}"))) };
        let kind_str = format!("{:?}", raw.kind());
        // Consider I-frames as keyframes.
        if kind_str.starts_with('I') {
            let ts_opt = raw.timestamp();
            let ts = video_rs::time::Time::new(ts_opt, time_base);
            let time = ts.as_secs_f64();
            out.push((index, time, kind_str, 0));
        }
        index += 1;
    }
    Ok(out)
}

fn load_impl<'py>(
    py: Python<'py>,
    input: &str,
) -> PyResult<(Bound<'py, PyArray4<u8>>, f64, f64, usize, usize, usize)> {
    let (bytes, frames, width, height, fps, duration) =
        videors_decode_all(input).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let frame_size = width * height * 3;
    if frame_size == 0 {
        return Err(PyRuntimeError::new_err("invalid frame size"));
    }
    if bytes.len() != frames * frame_size {
        return Err(PyRuntimeError::new_err("incomplete frame buffer"));
    }
    let array =
        ndarray::Array4::from_shape_vec((frames, height, width, 3usize), bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to build ndarray: {}", e)))?;
    let py_arr = PyArray4::from_owned_array(py, array);
    Ok((
        py_arr,
        fps,
        duration,
        width,
        height,
        frames,
    ))
}

#[pyfunction]
pub fn load_from_path<'py>(
    py: Python<'py>,
    path: &str,
) -> PyResult<(Bound<'py, PyArray4<u8>>, f64, f64, usize, usize, usize)> {
    load_impl(py, path)
}

#[pyfunction]
pub fn load_from_url<'py>(
    py: Python<'py>,
    url: &str,
) -> PyResult<(Bound<'py, PyArray4<u8>>, f64, f64, usize, usize, usize)> {
    load_impl(py, url)
}

#[pyfunction]
pub fn keyframes_from_path(path: &str) -> PyResult<Vec<(usize, f64, String, usize)>> {
    videors_keyframes(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
pub fn keyframes_from_url(url: &str) -> PyResult<Vec<(usize, f64, String, usize)>> {
    videors_keyframes(url).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

pub fn register_module(core_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let video_module = PyModule::new(core_module.py(), "video")?;
    video_module.add_function(wrap_pyfunction!(load_from_path, &video_module)?)?;
    video_module.add_function(wrap_pyfunction!(load_from_url, &video_module)?)?;
    video_module.add_function(wrap_pyfunction!(keyframes_from_path, &video_module)?)?;
    video_module.add_function(wrap_pyfunction!(keyframes_from_url, &video_module)?)?;
    core_module.add_submodule(&video_module)?;
    Ok(())
}
