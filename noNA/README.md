# noNA Folder Documentation

This folder contains files related to numerical anomaly detection for PyTorch GPU models under normal execution (without NaN injection).

## Main Execution Flow

### 1. Normal Model Test Execution
- **`test_all_models.py`** - Main script for testing all PyTorch models
  - Includes models: autoencoder, simple_gan, simple_resnet, mini_transformer, stock_lstm, spam_chatbot
  - Executes training and inference for each model under normal conditions
  - Monitors numerical anomalies through nixnan library
  - Generates **`test_results_pytorch_GPU_test_all_models_py`** raw trace file

### 2. Result Processing and Conversion
- **`test_results_pytorch_GPU_test_all_models_py`** → **`test_results_pytorch_GPU_test_all_models.md`** (using `process_all_traces.py`)
- **`test_results_pytorch_GPU_test_all_models.md`** → **`test_results_pytorch_GPU_test_all_models.html`** (using `convert_md_with_details.py`)

## File Descriptions

### Core Files
- **`test_all_models.py`** - Main test script that sequentially executes all models
- **`test_report.txt`** - Test execution report

### Output Files
- **`test_results_pytorch_GPU_test_all_models_py`** - Raw numerical anomaly trace records
- **`test_results_pytorch_GPU_test_all_models.md`** - Markdown format analysis result table
- **`test_results_pytorch_GPU_test_all_models.html`** - HTML format interactive analysis results
- **`detail_*.html`** - Detailed instruction information pages (separated when INSTs field is too long)

## test_results_pytorch_GPU_test_all_models.md Content Description

This file is a 4-column Markdown table that records numerical anomalies detected under normal execution conditions:

| Column | Description |
|--------|-------------|
| **Folder** | Test model folder name (e.g., autoencoder, simple_gan) |
| **NameOfKernel** | CUDA kernel name that produced anomalies |
| **FunctionsRun** | Set of functions called during execution |
| **INSTs** | Detailed information of instructions with detected numerical anomalies |

### Statistics
- **Processed 42 kernel records**
- Main anomaly types include subnormal (extremely small values)
- Covers multiple different machine learning models

## Comparison with withNA

### test_results_pytorch_GPU_test_all_models.md (noNA)
- **42 kernel records**
- Mainly detects naturally occurring numerical anomalies
- Anomalies are primarily subnormal (extremely small values) types
- Reflects numerical behavior of models under normal conditions

### trace_withNA_output.md (withNA)  
- **43 kernel records**
- Contains artificially injected NaN and their propagation effects
- More diverse anomaly types: nan, inf, subnormal, div0
- Shows model responses and propagation paths to NaN injection

### Key Differences
1. **Anomaly Source**:
   - noNA: Naturally occurring numerical issues during model computation
   - withNA: Cascading anomalies caused by artificially injected NaN

2. **Anomaly Type Distribution**:
   - noNA: Mainly subnormal, occasionally other types
   - withNA: Contains more nan and inf type anomalies

3. **Detection Count**:
   - noNA: Fewer anomaly detections, reflecting normal operation state
   - withNA: More anomaly detections, showing NaN propagation effects

4. **Analysis Value**:
   - noNA: Understanding baseline numerical stability of models
   - withNA: Evaluating model robustness and handling capability for numerical anomalies

These two datasets are complementary, providing a complete perspective on model numerical behavior.