# withNA Folder Documentation

This folder contains files related to numerical anomaly detection testing for PyTorch GPU models under NaN injection conditions.

## Main Execution Flow

### 1. NaN Injection Test Execution
- **`run_all_models_with_nan_reexec.py`** - Main execution script
  - Uses `NA_inject.py` to inject NaN values into all PyTorch models
  - Sets 5% element corruption probability, with all corruptions converted to NaN
  - Loads nixnan library through LD_PRELOAD for numerical anomaly monitoring
  - Executes `../noNA/test_all_models.py` and captures all output
  - Generates **`trace_withNA_output.txt`** raw trace file

### 2. Result Processing and Conversion
- **`trace_withNA_output.txt`** → **`trace_withNA_output.md`** (using `process_all_traces.py`)
- **`trace_withNA_output.md`** → **`trace_withNA_output.html`** (using `convert_md_with_details.py`)

## File Descriptions

### Core Files
- **`run_all_models_with_nan_reexec.py`** - NaN injection test executor
- **`NA_inject.py`** - NaN injection decorator for randomly injecting NaN values during model forward propagation

### Output Files
- **`trace_withNA_output.txt`** - Raw numerical anomaly trace records
- **`trace_withNA_output.md`** - Markdown format analysis result table
- **`trace_withNA_output.html`** - HTML format interactive analysis results
- **`detail_*.html`** - Detailed instruction information pages (separated when INSTs field is too long)

## run_all_models_with_nan_reexec.py Functionality Details

### Main Features
1. **Environment Setup**: Sets LD_PRELOAD environment variable to load nixnan.so library
2. **Self Re-execution**: Ensures nixnan library is loaded at program startup
3. **Output Redirection**: Redirects all stdout and stderr to trace file
4. **NaN Injection**: Uses decorator pattern to wrap all nn.Module.forward methods
5. **Test Execution**: Runs complete PyTorch model test suite

### Technical Details
```python
# NaN injection configuration
nn.Module.forward = inject(
    corruption_probability=0.05,    # 5% element corruption probability
    nan_frac=1.0                    # All corruptions become NaN
)(nn.Module.forward)
```

### Execution Flow
1. Check LD_PRELOAD environment variable, re-execute itself if not set
2. Open trace file and redirect stdout/stderr
3. Load NA_inject.py and wrap all model forward methods
4. Execute `../noNA/test_all_models.py`
5. Capture all nixnan output to `trace_withNA_output.txt`

## trace_withNA_output.md Content Description

This file is a 4-column Markdown table containing:

| Column | Description |
|--------|-------------|
| **Folder** | Test model folder name |
| **NameOfKernel** | CUDA kernel name |
| **FunctionsRun** | Set of functions called during execution |
| **INSTs** | Detailed information of instructions with detected numerical anomalies |

### INSTs Field Format
```
[['error_types'], operand_number, 'instruction_name']
```
- **error_types**: Anomaly types (e.g., subnormal, inf, nan)
- **operand_number**: Operand number where anomaly occurred
- **instruction_name**: CUDA instruction name that produced the anomaly

### Detected Anomaly Types
- **subnormal**: Extremely small numerical values
- **inf**: Infinite values
- **nan**: Not a Number values
- **div0**: Division by zero errors

## Differences from noNA

The withNA folder results demonstrate:
- Abnormal behavior of models after artificial NaN injection
- How NaN propagates through the computation graph
- Sensitivity of different CUDA kernels to NaN
- More numerical anomaly detections compared to noNA

This data helps understand model numerical stability and anomaly handling capabilities.