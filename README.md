# QCBench Evaluation Pipeline

This repository contains a complete pipeline for evaluating Large Language Models (LLMs) on the QCBench chemistry dataset using both traditional accuracy metrics and the xVerify verification system.

## Installation

First, install the required dependencies:

```bash
pip install datasets swanlab
```

## Files Overview

- `inference.py` - Runs LLM inference on QCBench dataset
- `eval.py` - Evaluates model answers using numerical accuracy metrics
- `xVerify_eval.py` - Evaluates model answers using xVerify verification system
- `report.py` - Generates comprehensive analysis reports
- `QCBench.json` - The chemistry dataset containing 350 questions

## Workflow

### Step 1: Run Inference

First, run the inference script to generate model answers:

```bash
python inference.py --model your_model_name --workers 30
```

**Important**: Before running, update the following in `inference.py`:
- Replace `your_url` with your API endpoint URL
- Replace `your_api_key` with your API key

The script will:
- Load the QCBench dataset (350 chemistry questions)
- Send questions to your specified LLM model
- Save results to `results/results_{model_name}.jsonl`

### Step 2: Evaluate Results

#### Option A: Traditional Accuracy Evaluation

Update the file paths in `eval.py`:
- Replace `your_result_path` with the path to your inference results file
- Replace the second `your_result_path` with your desired output path

Then run:
```bash
python eval.py
```

This will:
- Extract answers from `\boxed{}` LaTeX environments
- Compare numerical answers with ground truth
- Calculate accuracy scores
- Save scored results to JSON format

#### Option B: xVerify Evaluation

Update the input path in `xVerify_eval.py`:
- Replace the default input path with your inference results file path

Then run:
```bash
python xVerify_eval.py --input your_results_file.json
```

This will:
- Use the xVerify model to verify answer correctness
- Provide detailed verification scores
- Save verification results

### Step 3: Generate Reports

Update the input path in `report.py`:
- Replace the default input path with your evaluation results file path

Then run:
```bash
python report.py --input your_evaluation_results.json --model your_model_name
```

This will:
- Calculate overall accuracy
- Calculate per-class accuracy for each chemistry category
- Generate detailed performance reports
- Optionally log results to SwanLab (use `--no-swanlab` to disable)

## Dataset Categories

The QCBench dataset includes the following chemistry categories:
- **Analytical Chemistry**
- **Biochemistry** (merged with Organic as "BOC" in reports)
- **Inorganic Chemistry**
- **Materials Science**
- **Organic Chemistry** (merged with Biochemistry as "BOC" in reports)
- **Physical Chemistry**
- **Polymer Chemistry**
- **Technical Chemistry**

## Output Structure

```
results/
├── results_{model_name}.jsonl          # Raw inference results
├── results_{model_name}.json           # Scored results (from eval.py)
└── xverify_results/                    # xVerify evaluation results

reports/
└── report_{model_name}.json            # Analysis reports
```

## Example Usage

```bash
# 1. Run inference
python inference.py --model gpt-4o --workers 30

# 2. Evaluate accuracy
python eval.py

# 3. Generate report
python report.py --input data/results_acc/results_gpt-4o.json --model gpt-4o
```

## Configuration

### Model Configuration
- Update API endpoints and keys in `inference.py`
- Modify system prompts for different evaluation scenarios
- Adjust timeout and retry settings as needed

### Evaluation Settings
- Modify tolerance settings in `eval.py` for numerical comparisons
- Adjust xVerify model parameters in `xVerify_eval.py`
- Customize report generation options in `report.py`

## Notes

- The pipeline automatically handles Biochemistry and Organic chemistry categories by merging them as "BOC" in reports
- All numerical comparisons use high-precision decimal arithmetic to avoid floating-point errors
- The xVerify evaluation requires the xVerify model to be properly installed and configured
- SwanLab integration is optional and can be disabled with the `--no-swanlab` flag

## Troubleshooting

1. **API Connection Issues**: Check your API endpoint and key configuration in `inference.py`
2. **File Path Errors**: Ensure all file paths are correctly updated in each script
3. **xVerify Model Issues**: Verify xVerify model installation and configuration
4. **Memory Issues**: Reduce the number of workers in inference.py if encountering memory problems

## Dependencies

- `datasets` - For dataset loading utilities
- `swanlab` - For experiment tracking (optional)
- `requests` - For API calls
- `tqdm` - For progress bars
- `decimal` - For high-precision numerical operations
- `json` - For data serialization
- `os` - For file operations
- `argparse` - For command-line argument parsing 