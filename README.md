🎙️ **Speech-Generation-Project**

# Text-to-Audio Generation with Fine-Tuning & Hardware Optimization

## 1. Overview
This project explores Text-to-Audio (TTA) generation using advanced deep learning models, focusing on fine-tuning techniques and hardware efficiency. The research evaluates how GPUs and NPUs handle computational workloads for speech/audio synthesis, aiming to optimize both model quality and resource utilization.

Key areas include model parameters, audio features, and efficient fine-tuning strategies (LoRA, PEFT, quantization). The goal is to develop a framework for high-quality audio with reduced latency, memory usage, and energy consumption.

## 2. Objectives
- Analyze baseline performance of state-of-the-art Text-to-Audio models (FastSpeech2, VITS, HiFi-GAN).
- Study GPU utilization vs NPU utilization during training & inference.
- Understand model parameters and their impact on quality (hidden layers, embedding size, attention heads).
- Explore audio features (mel-spectrograms, pitch, prosody) for improved synthesis.
- Apply fine-tuning techniques:
	- LoRA (Low-Rank Adaptation)
	- PEFT (Parameter-Efficient Fine-Tuning)
	- Model Quantization (4-bit / 8-bit)
- Benchmark models across quality (MOS, WER) and efficiency (latency, throughput, energy usage).

## 3. Scope
- **Models:** FastSpeech2, VITS, HiFi-GAN
- **Fine-tuning:** LoRA, PEFT, QLoRA, quantization
- **Hardware:** NVIDIA GPU (A100/RTX 4090), NPU (Google Edge TPU, Qualcomm Hexagon, AWS Inferentia)
- **Datasets:**
	- LJ Speech (single-speaker)
	- LibriTTS (multi-speaker)
	- CommonVoice (optional)

## 4. Methodology
**Phase 1: Baseline Setup**
- Train and evaluate baseline models on GPUs
- Extract metrics: training time, GPU utilization, inference speed

**Phase 2: Fine-Tuning**
- Apply LoRA & PEFT to reduce trainable parameters
- Experiment with quantization for memory and inference improvements

**Phase 3: Hardware Benchmarking**
- Deploy fine-tuned models on GPUs vs NPUs
- Record utilization, latency, energy consumption

**Phase 4: Evaluation**
- Audio quality: Mean Opinion Score (MOS), Perceptual Evaluation of Speech Quality (PESQ)
- Efficiency: latency (ms), throughput, GPU/NPU utilization %, energy (watts)

## 5. Tools & Frameworks
- **Modeling:** PyTorch, TensorFlow
- **Fine-tuning:** Hugging Face PEFT, LoRA, QLoRA
- **Hardware Profiling:** NVIDIA Nsight, PyTorch Profiler, NPU SDKs
- **Evaluation:** librosa, PESQ, subjective MOS test
- **Deployment:** ONNX Runtime, TensorRT

## 6. Timeline (2 Months / 8 Weeks)
- **Week 1–2:** Baseline setup
	- Install dependencies, prepare datasets
	- Train baseline models (FastSpeech2, VITS)
	- Collect GPU utilization stats
- **Week 3–4:** Fine-tuning
	- Apply LoRA and PEFT methods
	- Experiment with quantization (8-bit, 4-bit)
	- Compare results with full fine-tuning
- **Week 5–6:** Hardware benchmarking
	- Deploy models on GPU & NPU
	- Record inference latency, throughput, energy use
	- Optimize pipelines with TensorRT
- **Week 7–8:** Evaluation & Documentation
	- Conduct MOS tests with human listeners
	- Summarize GPU vs NPU trade-offs
	- Write final research report & publish results

## 7. Expected Deliverables
- Baseline performance report for Text-to-Audio models
- Comparative analysis of GPU vs NPU performance
- Fine-tuning experiments (LoRA, PEFT, quantization) with results
- Audio quality evaluation dataset & results (MOS, PESQ)
- Final research paper/report with recommendations
- Open-source code repository for reproducibility