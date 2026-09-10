# ONNX Runtime Edge NLP Sequence Classification

This example demonstrates how to export a lightweight transformer model (such as a MiniLM or DistilBERT model) from PyTorch to the ONNX format and run low-latency inference on the edge using **ONNX Runtime**.

Running NLP models locally on edge devices (such as mobile platforms, laptops, or desktop clients) is a key design pattern for modern applications. It allows you to:
1. **Optimize Data Privacy**: Keystrokes and conversation text never leave the device.
2. **Eliminate Inference Costs**: Edge execution leverages local CPU/GPU resources rather than invoking billed cloud APIs.
3. **Achieve Near Zero-Latency**: Sub-15ms local inference fits comfortably within immediate UI rendering cycles (e.g. keypress events).

---

## Directory Structure

* **`export_model.py`**: Downloads a pre-trained sequence classification model from Hugging Face and exports it to the ONNX format using `torch.onnx.export` with dynamic axes.
* **`inference.py`**: Performs local inference using `onnxruntime` on input text.
* **`benchmark.py`**: Benchmarks initialization startup latency, disk/memory size, and execution speed.
* **`test_inference.py`**: Test suite validating model output labels and confidence levels.
* **`requirements.txt`**: Dependency requirements file.

---

## Setup Instructions

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Step-by-Step Usage

### 1. Export the Model to ONNX

Run the exporter script to compile the model to ONNX format. By default, it exports the emotion classification model `bhadresh-savani/distilbert-base-uncased-emotion`:

```bash
python export_model.py --output_path model.onnx
```

This exports `model.onnx` configured with **dynamic axes** (enabling variable batch sizes and sequence lengths).

### 2. Run Inference

Provide text directly in the command line or run interactively to classify sentiment/emotions and estimate intent context:

```bash
python inference.py --model model.onnx --text "I feel stressed and ignored today"
```

#### Example Output:
```text
Loading ONNX Runtime InferenceSession for 'model.onnx'...
Loading tokenizer: 'bhadresh-savani/distilbert-base-uncased-emotion'...

Running inference...

Input:
"I feel stressed and ignored today"

Output:
Emotion scores:
- sadness: 0.50
- anger: 0.49
- fear: 0.00
- joy: 0.00
- love: 0.00
- surprise: 0.00
- stress: 0.75
- neutral: 0.00

Intent:
- sharing
```

---

## Running Benchmarks

Profile the initialization latency and model execution speed over 100 runs:

```bash
python benchmark.py --model model.onnx --runs 100
```

### Benchmark Results (Apple Silicon M5 CPU)

| Metric | Measurement |
| :--- | :--- |
| **Model Size on Disk** | **0.88 MB** (Compiled/Exported ORT Graph) |
| **Session Initialization Latency** | **~138 ms** (Cold startup time) |
| **Average Execution Latency (p50)** | **~2.96 ms** |
| **99th Percentile Latency (p99)** | **~4.73 ms** |

---

## Running Tests

Verify the correctness of classification predictions and dynamic input bindings using `pytest`:

```bash
pytest test_inference.py
```
