# osc-data

多模态数据表示与处理库，提供 Python 友好的 API 和 Rust 实现的高性能核心。

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-2021-orange?logo=rust)](https://www.rust-lang.org/)

## 简介

`osc-data` 是一个用于多模态数据处理（文本、图像、音频、视频）的 Python 库，核心计算模块使用 Rust 实现以提供高性能。

## 特性

- **文本处理**：流式句子分割，支持中英文混合文本
- **图像处理**：加载、保存、格式转换、缩放、裁剪，支持 PNG/JPEG/WebP/BMP/GIF/TIFF
- **音频处理**：音频加载、特征提取（分贝计算等）
- **视频处理**：视频加载、帧提取、关键帧分割、音频提取与合并、保存
- **音视频同步**：音频时长自动调整（循环/静音填充）、音频截断警告
- **高性能**：核心算法使用 Rust + PyO3 实现，Python 端使用 DocArray 类型系统

## 安装

### 环境要求

- Python >= 3.9
- Rust >= 1.70 (编译时需要，仅源码安装需要)

### 使用 pip 安装

```bash
# 从 PyPI 安装 
pip install osc-data

# 从 Git 仓库直接安装
pip install git+https://github.com/username/osc-data.git

# 安装指定版本
pip install osc-data==0.2.8
```

### 从源码安装

```bash
# 克隆仓库
git clone <repository-url>
cd osc-data

# 使用 uv 安装 (推荐)
uv sync

# 或使用 pip + maturin 构建
pip install maturin
maturin develop --release

# 或以可编辑模式安装
pip install -e .
```

## 快速开始

### 图像处理

```python
from osc_data.image import Image

# 从本地或 URL 加载图片
img = Image(uri="./photo.jpg").load()
# img = Image(uri="https://example.com/image.png").load()

# 查看图片信息
print(f"尺寸: {img.width}x{img.height}, 色彩模式: {img.color_mode}")
print(f"源格式: {img.source_format}")  # png, jpeg 等

# 格式转换
rgb_img = img.to_rgb()

# 缩放 (使用 Lanczos3 高质量算法)
small_img = img.resize(256, 256)

# 裁剪
cropped = img.crop(100, 100, 200, 200)

# 保存 (使用 file_format 指定输出格式)
img.save("output.png")
img.save("output.jpg", file_format="jpeg", quality=95)

# 批量处理
images = [img1, img2, img3]
resized_batch = Image.batch_resize_images(images, 256, 256)
```

### 文本流处理

```python
from osc_data.text_stream import TextStreamSentencizer

sentencizer = TextStreamSentencizer()

# 流式处理文本
streaming_text = "你好！这是第一句话。这是第二句话。"

sentences = []
for char in streaming_text:
    sentences.extend(sentencizer.push(char))

# 输出剩余内容
sentences.extend(sentencizer.flush())

print(sentences)
# ['你好！', '这是第一句话。', '这是第二句话。']
```

### 视频处理

```python
from osc_data.video import Video
from osc_data.audio import Audio

# 加载视频（或使用示例视频）
video = Video(uri="./video.mp4").load()
# video = Video().load_example()  # 加载内置示例视频

print(f"分辨率: {video.width}x{video.height}, FPS: {video.fps}, 时长: {video.duration}s")

# 按关键帧分割
segments = video.split_by_key_frames(min_split_duration_s=5)

# 提取音频
audio = video.extract_audio()

# 合并音频（自动调整音频时长匹配视频）
new_audio = Audio(uri="./music.mp3").load()
merged = video.merge_audio(new_audio, "output.mp4", audio_mode="loop")
# audio_mode: "loop" 循环填充, "silence" 静音填充

# 移除音频
no_audio = video.remove_audio("silent.mp4")

# 静态方法合并视频和音频
combined = Video.combine_video_audio(video, audio, "final.mp4")

# 在 Jupyter 中显示视频
video.display()

# 根据宽高比计算最佳尺寸
best_w, best_h = video.get_best_size((9, 16))
print(f"9:16 最佳尺寸: {best_w}x{best_h}")

# 保存视频
video.save("output.mp4", format="mp4", codec="h264")
```

### 音频处理

```python
from osc_data.audio import Audio

# 加载音频
audio = Audio(uri="./audio.wav").load()

print(f"采样率: {audio.sampling_rate}, 时长: {audio.duration}s")

# 计算分贝
from osc_data._core import compute_decibel
db = compute_decibel(audio.data, audio.sampling_rate)
```

## API 文档

### Image 类

| 属性/方法 | 说明 |
|-----------|------|
| `uri` | 图片路径或 URL |
| `color_mode` | 色彩模式 (RGB/RGBA/L) |
| `source_format` | 源文件格式 (png/jpeg/webp) |
| `width`, `height` | 图片尺寸 |
| `load()` | 从本地路径或 URL 加载图片 |
| `save(path, file_format, quality)` | 保存图片到本地 |
| `to_rgb()` | 转换为 RGB 格式 |
| `resize(width, height)` | 缩放图片 (Lanczos3) |
| `crop(x, y, width, height)` | 裁剪图片 |
| `to_bytes(file_format, quality)` | 转换为字节 |
| `from_bytes(data)` | 从字节创建图片 |
| `batch_resize_images(images, width, height)` | 批量缩放 |
| `display()` | 显示图片 (Jupyter) |

### TextStreamSentencizer 类

| 方法 | 说明 |
|------|------|
| `push(char)` | 推入单个字符，返回已完成的句子列表 |
| `flush()` | 清空缓冲区，返回剩余内容 |
| `reset()` | 重置状态 |

### Video 类

| 属性/方法 | 说明 |
|-----------|------|
| `uri` | 视频路径或 URL |
| `width`, `height` | 视频分辨率 |
| `fps` | 帧率 |
| `duration` | 时长（秒） |
| `has_audio` | 是否有音轨 |
| `load()` | 加载视频 |
| `load_example()` | 加载内置示例视频 |
| `save(path, format, codec)` | 保存视频 |
| `display()` | 显示视频 (Jupyter) |
| `split_by_key_frames(min_split_duration_s)` | 按关键帧分割 |
| `extract_audio()` | 提取音频 |
| `merge_audio(audio, output_path, audio_mode)` | 合并音频（替换原音频） |
| `get_best_size(ratio)` | 根据宽高比计算最佳尺寸 |
| `remove_audio(output_path)` | 移除音频 |
| `combine_video_audio(video, audio, output_path)` | 静态方法：合并视频和音频 |

**音频时长处理**：
- `audio_mode="loop"`：循环播放音频填充（默认）
- `audio_mode="silence"`：静音填充
- 音频长于视频时自动截断并发出警告

## 项目结构

```
osc-data/
├── osc_data/           # Python 模块
│   ├── __init__.py
│   ├── image.py        # 图像处理
│   ├── video.py        # 视频处理
│   ├── audio.py        # 音频处理
│   ├── text.py         # 文本处理
│   ├── text_stream.py  # 流式文本分割
│   └── assets/         # 示例资源
│       ├── image/      # 示例图片
│       ├── audio/      # 示例音频
│       ├── video/      # 示例视频
│       └── text/       # 示例文本
├── src/                # Rust 核心模块
│   ├── lib.rs          # 模块入口
│   ├── image.rs        # 图像处理核心
│   ├── audio.rs        # 音频处理核心
│   ├── text.rs         # 文本处理核心
│   └── text_stream.rs  # 流式文本分割核心
├── tests/              # 测试文件
│   ├── test_image.py
│   ├── test_sentencizer.py
│   └── ...
├── Cargo.toml          # Rust 配置
├── pyproject.toml      # Python 配置
└── README.md           # 本文件
```

## 开发

### 构建 Rust 扩展

```bash
maturin develop        # 开发模式 (快速编译)
maturin develop --release  # 发布模式 (优化)
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_image.py -v
pytest tests/test_video.py -v
pytest tests/test_sentencizer.py -v
```

### 代码风格

```bash
# Python 代码格式化
ruff format osc_data/ tests/

# Rust 代码格式化
cargo fmt
```

## 依赖

### Python 依赖
- `pydantic` >= 2.11.7: 数据验证
- `docarray`: 多模态数据类型系统
- `numpy`: 数组操作
- `requests`: HTTP 请求
- `librosa` >= 0.11.0: 音频处理
- `av` >= 10.0.0: 视频/音频编解码
- `wasabi` >= 1.1.0: 格式化日志输出

### Rust 依赖
- `pyo3` >= 0.25.0: Python 绑定
- `numpy`: NumPy 数组操作
- `ndarray`: Rust 多维数组
- `image` >= 0.25.0: 图像处理
- `regex`: 正则表达式

## 作者

- **wangmengdi** - 790990241@qq.com

## License

[添加许可证信息]
