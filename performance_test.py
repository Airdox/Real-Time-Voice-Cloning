#!/usr/bin/env python3
"""Performance testing script for Real-Time Voice Cloning optimizations."""

import time
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from config import OptimizationConfig, AudioConfig
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from utils.default_models import ensure_default_models


class PerformanceTester:
    """Performance testing utilities."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        result = {
            'ram_mb': memory_info.rss / 1024 / 1024,
            'vram_mb': 0.0,
        }
        
        if torch.cuda.is_available():
            result['vram_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            
        return result
    
    def time_function(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Time a function execution and measure memory."""
        memory_before = self.measure_memory_usage()
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        memory_after = self.measure_memory_usage()
        
        return {
            'execution_time': end_time - start_time,
            'memory_before': memory_before,
            'memory_after': memory_after,
            'memory_delta_ram': memory_after['ram_mb'] - memory_before['ram_mb'],
            'memory_delta_vram': memory_after['vram_mb'] - memory_before['vram_mb'],
            'result': result
        }
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test model loading performance."""
        print("Testing model loading performance...")
        
        models_dir = Path("saved_models")
        ensure_default_models(models_dir)
        
        # Test encoder loading
        encoder_result = self.time_function(
            encoder.load_model, 
            models_dir / "default" / "encoder.pt"
        )
        
        # Test synthesizer loading
        synthesizer_result = self.time_function(
            Synthesizer,
            models_dir / "default" / "synthesizer.pt"
        )
        
        # Test vocoder loading
        vocoder_result = self.time_function(
            vocoder.load_model,
            models_dir / "default" / "vocoder.pt"
        )
        
        return {
            'encoder': encoder_result,
            'synthesizer': synthesizer_result,
            'vocoder': vocoder_result
        }
    
    def test_inference_performance(self, num_samples: int = 5) -> Dict[str, Any]:
        """Test inference performance with dummy data."""
        print(f"Testing inference performance with {num_samples} samples...")
        
        # Generate dummy audio (1 second at 16kHz)
        dummy_audio = np.random.randn(16000).astype(np.float32)
        dummy_text = "This is a test sentence for performance evaluation."
        
        results = []
        
        for i in range(num_samples):
            print(f"  Sample {i+1}/{num_samples}")
            
            # Test encoding
            embed_result = self.time_function(
                encoder.embed_utterance, 
                dummy_audio
            )
            
            # Test synthesis (get synthesizer instance)
            synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
            synthesis_result = self.time_function(
                synthesizer.synthesize_spectrograms,
                [dummy_text], [embed_result['result']]
            )
            
            # Test vocoding
            specs = synthesis_result['result']
            if specs and len(specs) > 0:
                mel = np.concatenate(specs, axis=1)
                vocoder_result = self.time_function(
                    vocoder.infer_waveform,
                    mel, target=200, overlap=50, progress_callback=lambda *args: None
                )
            else:
                vocoder_result = {'execution_time': 0, 'memory_delta_ram': 0, 'memory_delta_vram': 0}
            
            results.append({
                'embedding': embed_result,
                'synthesis': synthesis_result,
                'vocoding': vocoder_result,
                'total_time': (embed_result['execution_time'] + 
                             synthesis_result['execution_time'] + 
                             vocoder_result['execution_time'])
            })
            
            # Clear memory between iterations
            OptimizationConfig.optimize_memory_usage()
        
        # Calculate statistics
        total_times = [r['total_time'] for r in results]
        return {
            'samples': results,
            'stats': {
                'mean_time': np.mean(total_times),
                'std_time': np.std(total_times),
                'min_time': np.min(total_times),
                'max_time': np.max(total_times),
            }
        }
    
    def run_full_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance tests."""
        print("Starting comprehensive performance test...")
        
        # Apply optimizations
        OptimizationConfig.apply_torch_optimizations()
        
        # System info
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'ram_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            system_info.update({
                'gpu_name': gpu_props.name,
                'gpu_memory_gb': gpu_props.total_memory / 1024 / 1024 / 1024,
                'gpu_compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
            })
        
        print(f"System info: {system_info}")
        
        # Run tests
        loading_results = self.test_model_loading()
        inference_results = self.test_inference_performance()
        
        return {
            'system_info': system_info,
            'model_loading': loading_results,
            'inference_performance': inference_results,
            'optimization_config': {
                'memory_efficient': OptimizationConfig.ENABLE_MEMORY_EFFICIENT_MODE,
                'mixed_precision': OptimizationConfig.USE_MIXED_PRECISION,
                'cudnn_benchmark': OptimizationConfig.ENABLE_CUDNN_BENCHMARK,
            }
        }


def main():
    """Main function to run performance tests."""
    tester = PerformanceTester()
    results = tester.run_full_performance_test()
    
    print("\n" + "="*50)
    print("PERFORMANCE TEST RESULTS")
    print("="*50)
    
    # Print system info
    print(f"CPU cores: {results['system_info']['cpu_count']}")
    print(f"RAM: {results['system_info']['ram_total_gb']:.1f} GB")
    if results['system_info']['cuda_available']:
        print(f"GPU: {results['system_info']['gpu_name']}")
        print(f"GPU Memory: {results['system_info']['gpu_memory_gb']:.1f} GB")
    
    # Print model loading times
    print(f"\nModel Loading Times:")
    for model, result in results['model_loading'].items():
        if 'execution_time' in result:
            print(f"  {model}: {result['execution_time']:.2f}s")
    
    # Print inference performance
    stats = results['inference_performance']['stats']
    print(f"\nInference Performance (avg over samples):")
    print(f"  Mean time: {stats['mean_time']:.2f}s")
    print(f"  Std deviation: {stats['std_time']:.2f}s")
    print(f"  Min/Max: {stats['min_time']:.2f}s / {stats['max_time']:.2f}s")
    
    print("\nOptimizations applied:")
    for opt, enabled in results['optimization_config'].items():
        print(f"  {opt}: {'✓' if enabled else '✗'}")


if __name__ == "__main__":
    main()