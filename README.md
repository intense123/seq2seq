# Text-to-Python Code Generation using Seq2Seq Models

This project implements and compares three sequence-to-sequence models for translating English docstrings into Python code.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/intense123/seq2seq.git
cd seq2seq
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt')"

# Prepare data
python3 -c "from data.preprocess import prepare_data; prepare_data()"

# Train all models
python3 train.py --model all --num_epochs 10

# Evaluate
python3 evaluate.py --model all

# Visualize attention
python3 visualize_attention.py --num_examples 5
```

## Overview

The system translates natural language descriptions (docstrings) into functional Python code using:
1. **Vanilla RNN Seq2Seq** (Baseline) - BLEU-4: 0.0314
2. **LSTM Seq2Seq** (Improved) - BLEU-4: 0.0408 (+30%)
3. **LSTM with Bahdanau Attention** (Best) - BLEU-4: 0.0623 (+98%)

## Project Structure

```
text2code/
├── data/
│   ├── __init__.py
│   ├── preprocess.py      # Data loading and preprocessing
│   └── dataset.py         # PyTorch Dataset and DataLoader
├── models/
│   ├── __init__.py
│   ├── rnn_seq2seq.py     # Model 1: Vanilla RNN
│   ├── lstm_seq2seq.py    # Model 2: LSTM
│   └── lstm_attention.py  # Model 3: LSTM with Attention
├── train.py               # Training script
├── evaluate.py            # Evaluation script (BLEU, accuracy, etc.)
├── visualize_attention.py # Attention visualization
├── checkpoints/           # Saved model checkpoints
├── results/
│   ├── plots/            # Loss curves and attention heatmaps
│   └── metrics.json      # Evaluation metrics
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/intense123/seq2seq.git
cd seq2seq
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (for BLEU score):
```bash
python3 -c "import nltk; nltk.download('punkt')"
```

## Usage

**Note:** All commands should be run from the repository root directory.

### Step 1: Prepare Data

Download and preprocess the CodeSearchNet Python dataset:

```bash
python3 -c "from data.preprocess import prepare_data; prepare_data()"
```

This will:
- Download the CodeSearchNet dataset from Hugging Face
- Filter samples by length constraints (docstring ≤ 50 tokens, code ≤ 80 tokens)
- Create train/val/test splits (8000/1000/1000 samples)
- Build vocabularies
- Save processed data to `./data/`

### Step 2: Train Models

Train all three models:

```bash
python3 train.py --model all --num_epochs 20
```

Or train individual models:

```bash
# Train only RNN
python3 train.py --model rnn --num_epochs 20

# Train only LSTM
python3 train.py --model lstm --num_epochs 20

# Train only Attention
python3 train.py --model attention --num_epochs 20
```

**Training Options:**
- `--embed_dim`: Embedding dimension (default: 256)
- `--hidden_dim`: Hidden state dimension (default: 256)
- `--num_layers`: Number of RNN/LSTM layers (default: 1)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--teacher_forcing_ratio`: Teacher forcing probability (default: 0.5)
- `--num_epochs`: Number of training epochs (default: 20)

### Step 3: Evaluate Models

Evaluate all trained models on the test set:

```bash
python3 evaluate.py --model all
```

This computes:
- **Token-level accuracy**: Proportion of correctly predicted tokens
- **Exact match accuracy**: Proportion of perfectly matched sequences
- **BLEU scores**: BLEU-1, BLEU-2, BLEU-3, BLEU-4, and corpus BLEU
- **Performance by sequence length**: Analysis across different length ranges
- **Performance by docstring length**: Analysis by source sequence length

### Step 4: Visualize Attention

Generate attention heatmaps for the LSTM with Attention model:

```bash
python3 visualize_attention.py --num_examples 5
```

This creates:
- Attention heatmaps showing alignment between source and target tokens
- Analysis of attention patterns
- Interpretation of which source tokens influence target generation

### Step 5: Error Analysis (Optional)

Run detailed error analysis:

```bash
python3 error_analysis.py --model all
```

## Model Architectures

### Model 1: Vanilla RNN Seq2Seq
- **Encoder**: Simple RNN
- **Decoder**: Simple RNN
- **Context**: Fixed-length vector (final hidden state)
- **Limitations**: Struggles with long sequences, information bottleneck

### Model 2: LSTM Seq2Seq
- **Encoder**: LSTM
- **Decoder**: LSTM
- **Context**: Fixed-length vector (hidden + cell state)
- **Improvements**: Better long-term dependency modeling via gating

### Model 3: LSTM with Attention
- **Encoder**: Bidirectional LSTM
- **Decoder**: LSTM with Bahdanau attention
- **Context**: Dynamic, computed at each decoding step
- **Advantages**: No bottleneck, interpretable, best performance

## Configuration

All models use identical hyperparameters for fair comparison:
- Embedding dimension: 256
- Hidden dimension: 256
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss (ignoring padding)
- Teacher forcing ratio: 0.5
- Gradient clipping: 1.0

## Dataset

**Source**: CodeSearchNet Python dataset from Hugging Face
- **URL**: https://huggingface.co/datasets/Nan-Do/code-search-net-python
- **Input**: Docstrings (natural language descriptions)
- **Output**: Python function code
- **Preprocessing**: 
  - Docstrings: lowercase, whitespace tokenization
  - Code: case-sensitive, whitespace tokenization
  - Special tokens: `<sos>`, `<eos>`, `<pad>`, `<unk>`

**Data Splits**:
- Training: 8,000 samples
- Validation: 1,000 samples
- Testing: 1,000 samples

## Results

Results will be saved to `./results/`:
- `training_results.json`: Training and validation losses
- `evaluation_results.json`: Test set metrics for all models
- `plots/`: Loss curves and attention heatmaps
- `*_examples.txt`: Example predictions for each model
- `attention_analysis.txt`: Detailed attention pattern analysis

## Key Findings (Expected)

1. **RNN Baseline**: Establishes lower bound, struggles with longer sequences
2. **LSTM**: Significant improvement over RNN, better gradient flow
3. **LSTM + Attention**: Best performance, interpretable, handles variable-length sequences

## Evaluation Metrics

### Token Accuracy
Percentage of correctly predicted tokens (position-wise comparison).

### Exact Match
Percentage of sequences where prediction exactly matches reference.

### BLEU Score
Standard metric for sequence generation quality:
- BLEU-1: Unigram overlap
- BLEU-4: Up to 4-gram overlap
- Corpus BLEU: Overall dataset score

## Error Analysis

The evaluation script analyzes:
- Performance vs sequence length
- Common error patterns:
  - Syntax errors
  - Incorrect operators
  - Missing indentation
  - Wrong variable names
  - Incomplete logic

## Attention Visualization

For the attention model, heatmaps show:
- X-axis: Source tokens (docstring)
- Y-axis: Generated tokens (code)
- Color intensity: Attention weight

**Interpretation Example**:
> The token "maximum" in the docstring strongly attends to `max()` in the generated code, indicating semantic alignment.

## Reproducibility

Set random seed for reproducible results:
```bash
python train.py --seed 42
```

## Troubleshooting

**Out of Memory (OOM)**:
```bash
python3 train.py --model all --batch_size 16
```

**Slow Training**:
- Use GPU if available (automatically detected)
- Reduce dataset size by modifying `train_size` in data preparation

**Poor Results**:
```bash
python3 train.py --model all --num_epochs 30 --learning_rate 0.0005
```

**Module Not Found Errors**:
- Make sure you're in the repository root directory
- Activate your virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`

## Citation

Dataset:
```
@misc{codesearchnet,
  title={CodeSearchNet Challenge},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  year={2019}
}
```

## License

This project is for educational purposes as part of an academic assignment.

## Contact

For questions or issues, please refer to the course materials or contact the instructor.
