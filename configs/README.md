# WiFo Configuration Files

This directory contains YAML configuration files for WiFo training and evaluation.

## File Structure

```
configs/
├── base_training.yaml      # Paper baseline for base model
├── quick_test.yaml         # Fast iteration for development
└── evaluation/
    └── zero_shot.yaml     # Zero-shot evaluation config
```

## Usage

### Train with a config file

```bash
# Use paper baseline configuration
python src/train.py --config configs/base_training.yaml

# Use quick test configuration
python src/train.py --config configs/quick_test.yaml
```

### Override parameters with dot notation

```bash
# Override model size
python src/train.py --config configs/base.yaml --model.size tiny

# Override learning rate
python src/train.py --config configs/base.yaml --training.optimizer.lr 1e-3

# Override dataset
python src/train.py --config configs/base.yaml --data.dataset "DS1*D2"

# Multiple overrides
python src/train.py --config configs/base.yaml --model.size tiny --training.batch_size 64
```

### Evaluation with config

```bash
# Zero-shot evaluation
python src/eval.py --config configs/evaluation/zero_shot.yaml

# Override checkpoint path
python src/eval.py --config configs/evaluation/zero_shot.yaml --paths.checkpoint_path weights/lightning/wifo_tiny.ckpt
```

## Configuration Structure

The configuration is organized into logical sections:

- **experiment**: Metadata about the experiment
- **model**: Model architecture parameters
- **training**: Training hyperparameters (optimizer, scheduler, masking)
- **data**: Data loading and preprocessing
- **trainer**: PyTorch Lightning Trainer settings
- **paths**: File paths for inputs and outputs
- **system**: System settings (seed, device, process name)

All parameter overrides must use dot notation matching this structure.

## Creating Custom Configs

To create a custom configuration:

1. Copy an existing config as a template:
   ```bash
   cp configs/base_training.yaml configs/my_experiment.yaml
   ```

2. Edit the file to change the parameters you need

3. Use your custom config:
   ```bash
   python src/train.py --config configs/my_experiment.yaml
   ```

## Tips

- Start with `configs/quick_test.yaml` for fast development iteration
- Use `configs/base_training.yaml` to reproduce paper results
- All configs support the same dot notation overrides
- Config files are version-controlled, making experiments reproducible
