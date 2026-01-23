## 1. Project Setup

- [ ] 1.1 Add PyTorch Lightning to project dependencies (pyproject.toml or requirements.txt)
- [ ] 1.2 Verify PyTorch Lightning installation compatibility with existing PyTorch version
- [ ] 1.3 Create `test/` directory structure for unit tests

## 2. Lightning Module Implementation

- [ ] 2.1 Create `src/lightning_module.py` with `WiFoLightningModule` class
- [ ] 2.2 Implement `__init__` method to wrap existing WiFo model
- [ ] 2.3 Implement `configure_optimizers` method with AdamW and cosine LR scheduler
- [ ] 2.4 Implement `training_step` method with multi-strategy masking support
- [ ] 2.5 Implement `validation_step` method for NMSE computation
- [ ] 2.6 Implement `test_step` method for zero-shot evaluation
- [ ] 2.7 Implement `forward` method with noise injection (20 dB SNR)
- [ ] 2.8 Add logging hooks for TensorBoard integration (on_train_epoch_end, on_validation_epoch_end)

## 3. DataModule Implementation

- [ ] 3.1 Create `src/data_module.py` with `WiFoDataModule` class
- [ ] 3.2 Implement `setup` method to load .mat files and split data
- [ ] 3.3 Implement data preprocessing (normalization, complex-to-real conversion)
- [ ] 3.4 Implement `train_dataloader` method with shuffle=True
- [ ] 3.5 Implement `val_dataloader` method with shuffle=False
- [ ] 3.6 Implement `test_dataloader` method supporting multiple datasets
- [ ] 3.7 Add support for num_workers, pin_memory, and prefetch_factor

## 4. Training Entry Point

- [ ] 4.1 Create `src/train_lightning.py` as main training script
- [ ] 4.2 Implement argument parsing with argparse (matching existing main.py structure)
- [ ] 4.3 Initialize PyTorch Lightning Trainer with appropriate callbacks
- [ ] 4.4 Configure ModelCheckpoint callback for best model saving
- [ ] 4.5 Configure TensorBoard logger with appropriate log directory
- [ ] 4.6 Add optional Weights & Biases logger support
- [ ] 4.7 Implement training loop invocation with `trainer.fit()`
- [ ] 4.8 Implement test/evaluation loop with `trainer.test()`

## 5. Callbacks and Utilities

- [ ] 5.1 Create custom callback for masking strategy logging
- [ ] 5.2 Create custom callback for NMSE computation and result file generation
- [ ] 5.3 Add learning rate monitoring callback
- [ ] 5.4 Add gradient clipping configuration (clip_grad=0.05)

## 6. Unit Tests - Lightning Module

- [ ] 6.1 Create `test/test_lightning_module.py`
- [ ] 6.2 Test module initialization with different model sizes (tiny, little, small, base)
- [ ] 6.3 Test forward pass with random masking strategy
- [ ] 6.4 Test forward pass with temporal masking strategy
- [ ] 6.5 Test forward pass with frequency masking strategy
- [ ] 6.6 Test loss computation (MSE on masked patches only)
- [ ] 6.7 Test optimizer configuration (AdamW with weight decay)
- [ ] 6.8 Test learning rate scheduler (cosine with warmup)
- [ ] 6.9 Test noise injection (20 dB SNR)

## 7. Unit Tests - DataModule

- [ ] 7.1 Create `test/test_data_module.py`
- [ ] 7.2 Test .mat file loading and parsing
- [ ] 7.3 Test train/val/test data splitting (9k/1k/2k)
- [ ] 7.4 Test data normalization (per-dataset mean/variance)
- [ ] 7.5 Test complex-to-real tensor conversion
- [ ] 7.6 Test DataLoader creation (batch_size, num_workers, pin_memory)
- [ ] 7.7 Test multi-dataset support for zero-shot evaluation

## 8. Integration Tests

- [ ] 8.1 Create `test/test_training_integration.py`
- [ ] 8.2 Test complete training step (forward, loss, backward)
- [ ] 8.3 Test validation loop with NMSE computation
- [ ] 8.4 Test checkpoint saving and loading
- [ ] 8.5 Test multi-strategy training (random, temporal, frequency)
- [ ] 8.6 Test TensorBoard logging (verify log files created)
- [ ] 8.7 Test command-line argument parsing

## 9. Documentation

- [ ] 9.1 Add docstrings to all new classes and methods (Google style)
- [ ] 9.2 Create README section on PyTorch Lightning usage
- [ ] 9.3 Document migration guide from legacy TrainLoop to Lightning
- [ ] 9.4 Add example training commands to documentation

## 10. Validation and Testing

- [ ] 10.1 Run `openspec validate add-pytorch-lightning-training --strict`
- [ ] 10.2 Execute all unit tests and verify 100% pass rate
- [ ] 10.3 Run integration tests with a small sample dataset
- [ ] 10.4 Compare Lightning training results with legacy TrainLoop results (verify numerical equivalence)
- [ ] 10.5 Test on different model sizes (tiny, little, small, base)
- [ ] 10.6 Verify checkpoint compatibility (load Lightning checkpoint in legacy code if needed)

## Dependencies

- Task 2 depends on Task 1 (Lightning module requires dependencies installed)
- Task 4 depends on Task 2 and 3 (Training entry point requires module and datamodule)
- Task 6 depends on Task 2 (LightningModule tests require module implementation)
- Task 7 depends on Task 3 (DataModule tests require datamodule implementation)
- Task 8 depends on Task 2, 3, 4 (Integration tests require all components)
- Task 10 depends on all previous tasks (Validation requires complete implementation)

## Parallelizable Work

- Tasks 2 and 3 can be developed in parallel (Module and DataModule are independent)
- Tasks 6 and 7 can be developed in parallel (Test suites are independent)
- Tasks 9 can be done incrementally alongside implementation
