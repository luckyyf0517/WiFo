# Change: Add PyTorch Lightning Training Framework

## Why

The current WiFo training implementation uses a custom `TrainLoop` class with manual PyTorch training loops, which introduces boilerplate code, limited reproducibility features, and manual checkpoint management. Adopting PyTorch Lightning will provide a standardized, production-ready training framework with built-in features like automatic checkpointing, distributed training support, logging, and metrics tracking.

## What Changes

- **ADDED**: New PyTorch Lightning `LightningModule` wrapper for the WiFo model
- **ADDED**: PyTorch Lightning `LightningDataModule` for data loading and preprocessing
- **ADDED**: Training configuration using Lightning CLI or Trainer
- **ADDED**: Three-task training loop (random, temporal, frequency masking strategies) within Lightning framework
- **ADDED**: Comprehensive unit tests in `test/` directory
- **MODIFIED**: Training entry point to support both legacy and Lightning-based training
- **ADDED**: Callbacks for model checkpointing, early stopping, and learning rate monitoring
- **ADDED**: TensorBoard and optional Weights & Biases logging integration

## Impact

- **Affected specs**:
  - `pytorch-lightning-training` (NEW) - PyTorch Lightning-based training framework
- **Affected code**:
  - `src/train.py` - Keep as legacy, add reference to new Lightning module
  - `src/train_lightning.py` - NEW: PyTorch Lightning implementation
  - `src/data_module.py` - NEW: Lightning DataModule
  - `src/lightning_module.py` - NEW: LightningModule wrapper
  - `test/` - NEW: Unit tests for Lightning components

## Backward Compatibility

The existing `TrainLoop` class in `src/train.py` will be preserved for backward compatibility. Users can continue using the legacy training approach while the Lightning framework is introduced as an alternative, modernized training pathway.
