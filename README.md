# Real-Time Voice Cloning
This repository is an implementation of [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) with a vocoder that works in real-time. This was my [master's thesis](https://matheo.uliege.be/handle/2268.2/6801).

SV2TTS is a deep learning framework in three stages. In the first stage, one creates a digital representation of a voice from a few seconds of audio. In the second and third stages, this representation is used as reference to generate speech given arbitrary text.

**Video demonstration** (click the picture):

[![Toolbox demo](https://i.imgur.com/8lFUlgz.png)](https://www.youtube.com/watch?v=-O_hYhToKoA)



### Papers implemented  
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | This repo |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[1703.10135](https://arxiv.org/pdf/1703.10135.pdf) | Tacotron (synthesizer) | Tacotron: Towards End-to-End Speech Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification | This repo |

## Heads up
Like everything else in Deep Learning, this repo has quickly gotten old. Many SaaS apps (often paying) will give you a better audio quality than this repository will. If you wish for an open-source solution with a high voice quality:
- Check out [paperswithcode](https://paperswithcode.com/task/speech-synthesis/) for other repositories and recent research in the field of speech synthesis.
- Check out [Chatterbox](https://github.com/resemble-ai/chatterbox) for a similar project up to date with the 2025 SOTA in voice cloning

## Setup

### 1. Install Requirements
1. Both Windows and Linux are supported. A GPU is recommended for training and for inference speed, but is not mandatory.
2. Python 3.7 is recommended. Python 3.5 or greater should work, but you'll probably have to tweak the dependencies' versions. I recommend setting up a virtual environment using `venv`, but this is optional.
3. Install [ffmpeg](https://ffmpeg.org/download.html#get-packages). This is necessary for reading audio files.
4. Install [PyTorch](https://pytorch.org/get-started/locally/). Pick the latest stable version, your operating system, your package manager (pip by default) and finally pick any of the proposed CUDA versions if you have a GPU, otherwise pick CPU. Run the given command.
5. Install the remaining requirements with `pip install -r requirements.txt`

### 2. (Optional) Download Pretrained Models
Pretrained models are now downloaded automatically. If this doesn't work for you, you can manually download them [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models).

### 3. (Optional) Test Configuration
Before you download any dataset, you can begin by testing your configuration with:

`python demo_cli.py`

If all tests pass, you're good to go.

### 4. (Optional) Download Datasets
For playing with the toolbox alone, I only recommend downloading [`LibriSpeech/train-clean-100`](https://www.openslr.org/resources/12/train-clean-100.tar.gz). Extract the contents as `<datasets_root>/LibriSpeech/train-clean-100` where `<datasets_root>` is a directory of your choosing. Other datasets are supported in the toolbox, see [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Training#datasets). You're free not to download any dataset, but then you will need your own data as audio files or you will have to record it with the toolbox.

### 5. Launch the Toolbox
You can then try the toolbox:

`python demo_toolbox.py -d <datasets_root>`  
or  
`python demo_toolbox.py`  

depending on whether you downloaded any datasets. If you are running an X-server or if you have the error `Aborted (core dumped)`, see [this issue](https://github.com/CorentinJ/Real-Time-Voice-Cloning/issues/11#issuecomment-504733590).

## 🚀 Performance Optimizations

This repository has been enhanced with comprehensive performance optimizations:

### ⚡ Key Improvements

- **Updated Dependencies**: Modern, secure versions of all packages with compatibility ranges
- **Memory Optimization**: Efficient memory management with automatic GPU cache clearing
- **Performance Configuration**: Configurable optimization settings in `config.py`
- **Enhanced Error Handling**: Better file validation and user-friendly error messages
- **Modern Python Features**: Type hints, logging, and best practices throughout
- **Code Quality**: PEP8 compliance, Black formatting, and improved documentation

### 🔧 Performance Features

- **GPU Acceleration**: Optimized CUDA usage with mixed precision support
- **Memory Management**: Automatic memory optimization between inference steps
- **Batch Processing**: Configurable batch sizes for optimal performance
- **Audio Processing**: Optimized audio chunk processing with overlap handling
- **Model Loading**: Improved model loading with integrity verification

### 📊 Performance Testing

Run the performance test to evaluate optimizations on your system:

```bash
python performance_test.py
```

This will measure:
- Model loading times
- Inference performance
- Memory usage
- System capabilities

### ⚙️ Configuration

Customize performance settings in `config.py`:

```python
from config import OptimizationConfig, AudioConfig

# Enable optimizations
OptimizationConfig.apply_torch_optimizations()

# Configure for your hardware
config = OptimizationConfig.get_device_config()
```

### 🛠️ Development Setup

For development with enhanced tooling:

```bash
pip install -e .[dev]  # Install with development dependencies
black --line-length=100 .  # Format code
flake8 --max-line-length=100 .  # Lint code
```
