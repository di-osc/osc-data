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

fn videors_decode_rgb(input: &str, expected_width: usize, expected_height: usize) -> Result<(Vec<u8>, usize)> {
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

    for res in decoder.decode_iter() {
        let (_ts, frame) = match res {
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
        if expected_width != 0 && expected_height != 0 {
            if w != expected_width || h != expected_height {
                return Err(anyhow!("frame size mismatch with metadata"));
            }
        }
        if let Some(slice) = frame.as_slice() {
            bytes.extend_from_slice(slice);
        } else {
            let owned = frame.to_owned();
            bytes.extend_from_slice(owned.as_slice().ok_or_else(|| anyhow!("failed to get owned slice"))?);
        }
        frames += 1;
    }

    if width == 0 || height == 0 { return Err(anyhow!("no frames decoded")); }
    Ok((bytes, frames))
}

// ffmpeg CLI path removed in favor of video-rs

fn ffprobe_keyframes(input: &str, fps: f64) -> Result<Vec<(usize, f64, String, usize)>> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_frames")
        .arg("-show_entries")
        .arg("frame=key_frame,pict_type,pkt_pts_time,best_effort_timestamp_time,pkt_size")
        .arg("-of")
        .arg("json")
        .arg(input)
        .output()
        .with_context(|| "failed to execute ffprobe for frames")?;
    if !output.status.success() {
        return Err(anyhow!(
            "ffprobe frames failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let parsed: FfprobeFrames = serde_json::from_slice(&output.stdout)
        .with_context(|| "failed to parse ffprobe frames json")?;

    let mut out: Vec<(usize, f64, String, usize)> = Vec::new();
    for f in parsed.frames.into_iter() {
        if f.key_frame.unwrap_or(0) != 1 {
            continue;
        }
        let time_str = f
            .pkt_pts_time
            .as_ref()
            .or(f.best_effort_timestamp_time.as_ref())
            .cloned();
        let time: f64 = match time_str {
            Some(s) => s.parse::<f64>().unwrap_or(0.0),
            None => 0.0,
        };
        let approx_index = (time * fps).round() as usize;
        let pict = f.pict_type.unwrap_or_else(|| "?".to_string());
        let size = f
            .pkt_size
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        out.push((approx_index, time, pict, size));
    }
    Ok(out)
}

fn load_impl<'py>(
    py: Python<'py>,
    input: &str,
) -> PyResult<(Bound<'py, PyArray4<u8>>, f64, f64, usize, usize, usize)> {
    let meta = ffprobe_meta(input).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (bytes, frames) = videors_decode_rgb(input, meta.width, meta.height)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let frame_size = meta.width * meta.height * 3;
    if frame_size == 0 {
        return Err(PyRuntimeError::new_err("invalid frame size"));
    }
    if bytes.len() != frames * frame_size {
        return Err(PyRuntimeError::new_err("incomplete frame buffer"));
    }
    let array =
        ndarray::Array4::from_shape_vec((frames, meta.height, meta.width, 3usize), bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to build ndarray: {}", e)))?;
    let py_arr = PyArray4::from_owned_array(py, array);
    Ok((
        py_arr,
        meta.fps,
        meta.duration,
        meta.width,
        meta.height,
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
    let meta = ffprobe_meta(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    ffprobe_keyframes(path, meta.fps).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
pub fn keyframes_from_url(url: &str) -> PyResult<Vec<(usize, f64, String, usize)>> {
    let meta = ffprobe_meta(url).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    ffprobe_keyframes(url, meta.fps).map_err(|e| PyRuntimeError::new_err(e.to_string()))
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
