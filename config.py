"""Configuration settings for Real-Time Voice Cloning optimization."""

import torch
from typing import Dict, Any


# Performance optimization settings
class OptimizationConfig:
    """Configuration for performance optimizations."""

    # Memory optimization
    ENABLE_MEMORY_EFFICIENT_MODE = True
    CLEAR_CACHE_BETWEEN_INFERENCE = True

    # GPU optimization
    USE_MIXED_PRECISION = True
    ENABLE_CUDNN_BENCHMARK = True

    # Batch processing
    MAX_BATCH_SIZE = 4
    OPTIMAL_BATCH_SIZE = 2

    # Model optimization
    ENABLE_MODEL_COMPILATION = True  # For PyTorch 2.0+
    USE_SCRIPTING = False  # TorchScript optimization

    # Audio processing optimization
    AUDIO_CHUNK_SIZE = 16000  # 1 second at 16kHz
    OVERLAP_SIZE = 4000  # 0.25 seconds overlap

    @classmethod
    def apply_torch_optimizations(cls) -> None:
        """Apply PyTorch-specific optimizations."""
        if cls.ENABLE_CUDNN_BENCHMARK and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)

    @classmethod
    def get_device_config(cls) -> Dict[str, Any]:
        """Get optimal device configuration."""
        config = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "mixed_precision": cls.USE_MIXED_PRECISION and torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            config.update(
                {
                    "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
                    "compute_capability": torch.cuda.get_device_properties(0).major,
                }
            )

        return config

    @classmethod
    def optimize_memory_usage(cls) -> None:
        """Optimize memory usage."""
        if torch.cuda.is_available() and cls.CLEAR_CACHE_BETWEEN_INFERENCE:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# Audio quality settings
class AudioConfig:
    """Configuration for audio processing quality."""

    # Sampling rates
    ENCODER_SAMPLING_RATE = 16000
    SYNTHESIZER_SAMPLING_RATE = 22050
    VOCODER_SAMPLING_RATE = 22050

    # Quality settings
    MEL_CHANNELS = 80
    N_FFT = 2048
    HOP_LENGTH = 275
    WIN_LENGTH = 1100

    # Voice activity detection
    VAD_THRESHOLD = 0.5
    MIN_SPEECH_DURATION = 0.5  # seconds

    @classmethod
    def get_audio_config(cls) -> Dict[str, Any]:
        """Get audio processing configuration."""
        return {
            "encoder_sr": cls.ENCODER_SAMPLING_RATE,
            "synthesizer_sr": cls.SYNTHESIZER_SAMPLING_RATE,
            "vocoder_sr": cls.VOCODER_SAMPLING_RATE,
            "mel_channels": cls.MEL_CHANNELS,
            "n_fft": cls.N_FFT,
            "hop_length": cls.HOP_LENGTH,
            "win_length": cls.WIN_LENGTH,
            "vad_threshold": cls.VAD_THRESHOLD,
            "min_speech_duration": cls.MIN_SPEECH_DURATION,
        }
