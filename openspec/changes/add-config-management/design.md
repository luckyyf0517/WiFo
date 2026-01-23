# Design: Configuration Management System

## Overview

A hierarchical YAML-based configuration system that supports:
1. Loading configurations from YAML files
2. Merging with CLI argument overrides
3. Nested parameter access via dot notation (`--model.size base`)
4. Validation of required parameters
5. Type conversion and preservation

## Architecture

```
configs/                    # Configuration files directory
├── base_training.yaml      # Paper baseline configuration
├── quick_test.yaml         # Fast iteration configuration
├── evaluation/             # Evaluation configs
│   ├── zero_shot.yaml
│   └── full_benchmark.yaml
└── models/                 # Model-specific configs
    ├── tiny.yaml
    ├── base.yaml
    └── large.yaml

src/config/                 # Configuration module
├── __init__.py
├── loader.py              # YAML loading and parsing
├── parser.py              # CLI override parser
└── validator.py           # Configuration validation
```

## Configuration Schema

### Top-Level Structure

```yaml
# Experiment metadata
experiment:
  name: "wifo_base_pretraining"
  tags: ["pretraining", "base"]
  note: "Paper baseline configuration"

# Model architecture
model:
  size: "base"              # tiny, little, small, base
  patch_size: 4
  t_patch_size: 2
  pos_emb: "SinCos"
  no_qkv_bias: false
  conv_num: 3

# Training configuration
training:
  # Optimization
  optimizer:
    name: "adamw"
    lr: 5e-4
    min_lr: 1e-5
    weight_decay: 0.05
    betas: [0.9, 0.999]

  # Scheduler
  scheduler:
    name: "cosine"
    warmup_epochs: 5
    total_epochs: 200

  # Training loop
  batch_size: 128
  gradient_clip: 0.05
  early_stop: 5

  # Masking strategies
  mask:
    strategy: "random"      # random, temporal, frequency
    strategy_mode: "batch"  # batch, none
    ratio: 0.5              # default ratio when not in batch mode

# Data configuration
data:
  dataset: "DS1"
  data_path: "dataset/"
  train_split: 9000
  val_split: 1000
  test_split: 2000
  num_workers: 32
  pin_memory: true
  prefetch_factor: 4

# Lightning trainer settings
trainer:
  accelerator: "auto"
  devices: "auto"
  precision: "32-true"
  max_epochs: 200
  check_val_every_n_epoch: 1
  log_every_n_steps: 10

# Paths and outputs
paths:
  output_dir: "./experiments"
  log_dir: "./logs"
  checkpoint_path: ""      # for loading pre-trained weights

# System settings
system:
  seed: 100
  device_id: "0"
  process_name: "wifo_training"
```

## CLI Override Syntax

### Dot Notation (Required)

ALL parameter overrides MUST use dot notation matching the YAML structure:

```bash
# Override learning rate
python src/train.py --config configs/base.yaml --training.optimizer.lr 1e-3

# Override model size
python src/train.py --config configs/base.yaml --model.size tiny

# Override nested mask ratio
python src/train.py --config configs/base.yaml --training.mask.ratio 0.75

# Override dataset
python src/train.py --config configs/base.yaml --data.dataset DS2
```

**Important**: No shortcut arguments are provided. All overrides must use the full dot notation to ensure clarity and prevent ambiguity. This makes the configuration system predictable and easy to understand.

## Implementation Details

### Configuration Loading Flow

1. **Load YAML**: Parse YAML file into nested dictionary
2. **Apply Defaults**: Merge with hardcoded defaults for missing values
3. **Apply CLI Overrides**: Parse `--key.value` arguments and update config
4. **Validate**: Check required parameters and type constraints
5. **Flatten**: Convert nested dict to flat namespace for argparse compatibility

### Type Conversion

```python
# YAML types are preserved
lr: 5e-4              # -> float
batch_size: 128       # -> int
enable_logging: true  # -> bool
tags: ["a", "b"]       # -> list

# CLI overrides use type inference
--training.lr 1e-3     # -> float
--model.size tiny      # -> string
--data.num_workers 4   # -> int
```

### Backward Compatibility

When no `--config` is provided, the system falls back to the original argparse behavior:
- All existing CLI arguments continue to work
- Defaults are loaded from the same hardcoded values
- No breaking changes to existing workflows

## Example Configurations

### Paper Baseline (configs/base_training.yaml)

```yaml
experiment:
  name: "paper_baseline"
  note: "Reproduces paper results for base model"

model:
  size: "base"
  patch_size: 4
  t_patch_size: 4
  pos_emb: "SinCos"

training:
  optimizer:
    lr: 5e-4
    min_lr: 1e-5
    weight_decay: 0.05
  scheduler:
    warmup_epochs: 5
    total_epochs: 200
  batch_size: 128
  gradient_clip: 0.05
  mask:
    strategy_mode: "batch"  # Use all three strategies

data:
  dataset: "D1*D2*D3*D4*D5*D6*D7*D8*D9*D10*D11*D12*D13*D14*D15*D16"
```

### Quick Test (configs/quick_test.yaml)

```yaml
experiment:
  name: "quick_test"
  note: "Fast iteration with tiny model"

model:
  size: "tiny"

training:
  optimizer:
    lr: 1e-3
  scheduler:
    total_epochs: 10
  batch_size: 64

data:
  dataset: "DS1"
  num_workers: 4
```

## Validation Rules

### Required Parameters

- `model.size`: Must be one of [tiny, little, small, base]
- `data.dataset`: Must not be empty
- `training.optimizer.lr`: Must be positive float
- `training.batch_size`: Must be positive integer

### Type Constraints

```python
VALIDATION_RULES = {
    "model.size": lambda x: x in ["tiny", "little", "small", "base"],
    "training.optimizer.lr": lambda x: isinstance(x, float) and x > 0,
    "training.batch_size": lambda x: isinstance(x, int) and x > 0,
    # ... more rules
}
```

## Error Handling

1. **Config file not found**: Clear error with suggested config paths
2. **Invalid YAML**: Show line number and parsing error
3. **Invalid override**: Show the invalid key and valid alternatives
4. **Type mismatch**: Attempt conversion, show error if failed
5. **Missing required**: List all missing required parameters

## Future Extensions

1. **Config merging**: `--config base.yaml --config overrides.yaml`
2. **Config inheritance**: YAML extends keyword
3. **Config validation schema**: JSON Schema for strict validation
4. **Config generation**: Auto-generate config from saved checkpoint
5. **Config versioning**: Track config version in checkpoints
