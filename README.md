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

## 8. Environment Setup

To set up the Python environment for this project, follow these steps:

1. **Install Python**:
   - Ensure Python 3.8 or later is installed on your system. You can download it from [python.org](https://www.python.org/).

2. **Install Poetry**:
   - Poetry is used for dependency management and packaging.
   - Install Poetry by running:
     ```bash
     curl -sSL https://install.python-poetry.org | python3 -
     ```
   - Verify the installation:
     ```bash
     poetry --version
     ```

3. **Set Up the Environment**:
   - Navigate to the project directory:
     ```bash
     cd Speech-Generation-Project
     ```
   - Install dependencies:
     ```bash
     poetry install
     ```

4. **Activate the Virtual Environment**:
   - To activate the virtual environment created by Poetry, run:
     ```bash
     poetry shell
     ```

5. **Run the Project**:
   - After activating the environment, you can run the project scripts or notebooks as needed.

6. **Additional Tools**:
   - If you prefer using `venv` instead of Poetry, you can create a virtual environment manually:
     ```bash
     python3 -m venv env
     source env/bin/activate
     pip install -r requirements.txt
     ```

## 9. Text-to-Speech Details

This project includes a Text-to-Speech (TTS) pipeline for generating audio reviews from text. Below are the details of the implementation:

### Overview
The `product_reviews_generator.ipynb` notebook demonstrates how to convert a product review script into audio using the `Kokoro` library. The example provided focuses on a MacBook Air review, split into sections for better synthesis and playback.

### Key Features
- **TTS Pipeline**: Utilizes the `KPipeline` class from the `Kokoro` library to generate speech.
- **Audio Processing**: Converts text sections into audio files and combines them into a single audio file.
- **Languages and Voices**: Supports synthetic English pronunciation with an Afrikaans-like voice (`af_heart`).
- **Output**: Generates individual audio files for each section and a combined audio file for the entire review.

### Steps in the Notebook
1. **Install Dependencies**:
   - Installs the `kokoro` library and `soundfile` for audio processing.
   - Installs `espeak-ng` for text-to-speech synthesis.

2. **Initialize TTS Pipeline**:
   - Sets up the `KPipeline` with the desired language code and voice.

3. **Generate Audio Files**:
   - Loops through each section of the review script.
   - Synthesizes audio for each section and saves it as a `.wav` file.

4. **Combine Audio Files**:
   - Reads and concatenates all generated audio files.
   - Saves the combined audio as a single `.wav` file.

### Example Output
- Individual audio files: `macbook_review_part1_1.wav`, `macbook_review_part2_1.wav`, etc.
- Combined audio file: `macbook_review_combined.wav`

### Tools Used
- **Libraries**: `Kokoro`, `soundfile`, `numpy`
- **Audio Playback**: `IPython.display.Audio`
- **File Handling**: `os` for managing audio files

### How to Run
1. Open the `product_reviews_generator.ipynb` notebook.
2. Install the required dependencies.
3. Execute the cells to generate and combine audio files.
4. Listen to the generated audio files directly in the notebook or save them for external use.

## 10. Project Structure

This project is organized to facilitate easy navigation and development. Below is the recommended structure when using Poetry or `venv` for managing dependencies:

```
Speech-Generation-Project/
│
├── product_reviews_generator.ipynb  # Notebook for TTS pipeline
├── README.md                        # Project documentation
├── pyproject.toml                   # Poetry configuration file
├── poetry.lock                      # Poetry lock file for dependencies
├── requirements.txt                 # Optional: Dependencies for `venv`
├── .gitignore                       # Git ignore file
├── data/                            # Directory for datasets
│   ├── lj_speech/                   # LJ Speech dataset
│   ├── libri_tts/                   # LibriTTS dataset
│   └── common_voice/                # CommonVoice dataset (optional)
├── models/                          # Directory for saved models
│   ├── baseline/                    # Baseline models
│   └── fine_tuned/                  # Fine-tuned models
├── outputs/                         # Directory for generated outputs
│   ├── audio/                       # Generated audio files
│   └── logs/                        # Logs for training and evaluation
├── scripts/                         # Utility scripts
│   ├── preprocess.py                # Data preprocessing script
│   ├── train.py                     # Training script
│   └── evaluate.py                  # Evaluation script
└── tests/                           # Unit tests
    ├── test_pipeline.py             # Tests for TTS pipeline
    └── test_utils.py                # Tests for utility functions
```

### Key Files and Directories
- **`pyproject.toml`**: Defines the project dependencies and metadata for Poetry.
- **`requirements.txt`**: Lists dependencies for those using `venv` instead of Poetry.
- **`data/`**: Contains datasets used for training and evaluation.
- **`models/`**: Stores baseline and fine-tuned models.
- **`outputs/`**: Includes generated audio files and logs.
- **`scripts/`**: Contains Python scripts for preprocessing, training, and evaluation.
- **`tests/`**: Includes unit tests to ensure code reliability.

### Setting Up the Project
1. **Using Poetry**:
   - Install dependencies:
     ```bash
     poetry install
     ```
   - Run scripts:
     ```bash
     poetry run python scripts/train.py
     ```

2. **Using `venv`**:
   - Create a virtual environment:
     ```bash
     python3 -m venv env
     source env/bin/activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Run scripts:
     ```bash
     python scripts/train.py
     ```