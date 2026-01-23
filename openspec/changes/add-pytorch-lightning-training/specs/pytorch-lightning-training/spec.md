## ADDED Requirements

### Requirement: PyTorch Lightning Module Wrapper

The system SHALL provide a PyTorch Lightning `LightningModule` class that wraps the existing WiFo model architecture and implements the training forward pass, loss computation, and optimization steps.

#### Scenario: Module initialization
- **GIVEN** valid model configuration parameters (embed_dim, depth, decoder_depth, num_heads, etc.)
- **WHEN** initializing the WiFo Lightning module
- **THEN** the module SHALL create the underlying WiFo model with the specified architecture
- **AND** the module SHALL initialize AdamW optimizer with weight decay of 0.05
- **AND** the module SHALL configure cosine learning rate scheduler with 5-epoch warmup

#### Scenario: Training step with random masking
- **GIVEN** a batch of CSI tensors and mask_strategy='random'
- **WHEN** calling the training_step method
- **THEN** the module SHALL apply 85% random masking ratio
- **THEN** the module SHALL compute MSE loss on masked patches only
- **AND** the module SHALL return the loss value for backpropagation

#### Scenario: Training step with temporal masking
- **GIVEN** a batch of CSI tensors and mask_strategy='temporal'
- **WHEN** calling the training_step method
- **THEN** the module SHALL apply 50% temporal masking (future time steps)
- **THEN** the module SHALL compute MSE loss on masked patches only
- **AND** the module SHALL return the loss value for backpropagation

#### Scenario: Training step with frequency masking
- **GIVEN** a batch of CSI tensors and mask_strategy='frequency'
- **WHEN** calling the training_step method
- **THEN** the module SHALL apply 50% frequency masking (upper frequency bands)
- **THEN** the module SHALL compute MSE loss on masked patches only
- **AND** the module SHALL return the loss value for backpropagation

#### Scenario: Multi-strategy batch training
- **GIVEN** a batch of data and mask_strategy_random='batch'
- **WHEN** calling the training_step method
- **THEN** the module SHALL randomly select one of three masking strategies (random, temporal, frequency)
- **AND** the module SHALL apply the corresponding masking ratio
- **AND** the module SHALL log the selected strategy to TensorBoard

### Requirement: Lightning DataModule

The system SHALL provide a PyTorch Lightning `LightningDataModule` class that handles data loading, preprocessing, and train/validation/test splitting for WiFo training.

#### Scenario: DataModule setup
- **GIVEN** dataset configuration (path, batch_size, num_workers)
- **WHEN** calling the setup method
- **THEN** the DataModule SHALL load .mat files from the specified dataset paths
- **AND** the DataModule SHALL split data into 9k train / 1k val / 2k test samples per dataset
- **AND** the DataModule SHALL normalize CSI data using per-dataset mean and variance
- **AND** the DataModule SHALL convert complex tensors to real-valued tensors (real + imaginary channels)

#### Scenario: Train dataloader creation
- **GIVEN** a configured DataModule with training data
- **WHEN** calling the train_dataloader method
- **THEN** the method SHALL return a DataLoader with shuffle=True
- **AND** the DataLoader SHALL use the specified batch_size (default 128)
- **AND** the DataLoader SHALL support pin_memory and prefetch_factor for GPU optimization

#### Scenario: Validation dataloader creation
- **GIVEN** a configured DataModule with validation data
- **WHEN** calling the val_dataloader method
- **THEN** the method SHALL return a DataLoader with shuffle=False
- **AND** the DataLoader SHALL use the specified batch_size

#### Scenario: Test dataloader creation
- **GIVEN** a configured DataModule with test data and multiple datasets
- **WHEN** calling the test_dataloader method
- **THEN** the method SHALL return a list of DataLoaders, one per dataset
- **AND** each DataLoader SHALL have shuffle=False
- **AND** the DataLoaders SHALL support zero-shot evaluation on unseen datasets

### Requirement: Noise Injection During Training

The system SHALL inject complex Gaussian noise with 20 dB SNR into all CSI samples during both training and inference, as specified in the training documentation.

#### Scenario: Training noise injection
- **GIVEN** a batch of clean CSI tensors
- **WHEN** calling the forward pass during training
- **THEN** the module SHALL compute noise power based on signal power and 20 dB SNR
- **AND** the module SHALL add complex Gaussian noise to the input
- **AND** the loss SHALL be computed on the noisy input (denoising objective)

#### Scenario: Inference noise injection
- **GIVEN** a batch of clean CSI tensors during validation/test
- **WHEN** calling the forward pass during inference
- **THEN** the module SHALL compute noise power based on signal power and 20 dB SNR
- **AND** the module SHALL add complex Gaussian noise to the input
- **AND** the prediction SHALL be evaluated against the noisy target

### Requirement: Learning Rate Scheduling

The system SHALL implement cosine learning rate decay with warmup as specified in the training documentation.

#### Scenario: Warmup phase
- **GIVEN** a model with base learning rate of 5e-4 and warmup_epochs=5
- **WHEN** training epoch < 5
- **THEN** the learning rate SHALL linearly increase from min_lr to base_lr
- **AND** the learning rate SHALL be logged to TensorBoard each epoch

#### Scenario: Cosine decay phase
- **GIVEN** a model with base learning rate of 5e-4 and total_epochs=200
- **WHEN** training epoch >= 5
- **THEN** the learning rate SHALL follow cosine decay schedule
- **AND** the learning rate SHALL decay from base_lr to min_lr (1e-5) by epoch 200
- **AND** the learning rate SHALL be logged to TensorBoard each epoch

### Requirement: Model Checkpointing

The system SHALL provide automatic model checkpointing functionality through PyTorch Lightning callbacks.

#### Scenario: Best model checkpointing
- **GIVEN** a Trainer with ModelCheckpoint callback configured to monitor validation NMSE
- **WHEN** validation NMSE improves
- **THEN** the system SHALL save the model checkpoint to `experiments/{run_name}/model_save/best.ckpt`
- **AND** the checkpoint SHALL contain model state, optimizer state, and epoch number

#### Scenario: Epoch checkpointing
- **GIVEN** a Trainer with ModelCheckpoint callback configured to save every N epochs
- **WHEN** epoch % N == 0
- **THEN** the system SHALL save a checkpoint file to `experiments/{run_name}/model_save/epoch_{epoch}.ckpt`

### Requirement: Evaluation Metrics

The system SHALL compute Normalized Mean Square Error (NMSE) as the primary evaluation metric for model performance.

#### Scenario: NMSE computation for single dataset
- **GIVEN** model predictions and ground truth CSI tensors
- **WHEN** computing evaluation metrics
- **THEN** the system SHALL compute NMSE = mean(|H_pred - H_true|^2) / mean(|H_true|^2)
- **AND** the NMSE SHALL be computed only on masked patches
- **AND** the NMSE SHALL be logged to TensorBoard

#### Scenario: Multi-dataset zero-shot evaluation
- **GIVEN** a list of test datasets (e.g., D17, D18, D19)
- **WHEN** running zero-shot evaluation
- **THEN** the system SHALL compute NMSE for each dataset independently
- **AND** the system SHALL compute average NMSE across all datasets
- **AND** results SHALL be saved to `experiments/{run_name}/result.txt`

### Requirement: Logging and Monitoring

The system SHALL integrate with TensorBoard for training visualization and optionally support Weights & Biases.

#### Scenario: TensorBoard logging
- **GIVEN** a Trainer with TensorBoardLogger configured
- **WHEN** training progress occurs
- **THEN** the system SHALL log training loss each step
- **AND** the system SHALL log validation NMSE each epoch
- **AND** the system SHALL log learning rate each epoch
- **AND** the system SHALL log gradient norms each step (if grad_clip is enabled)

#### Scenario: Weights & Biases integration (optional)
- **GIVEN** a Trainer with WandBLogger configured and API key set
- **WHEN** training progress occurs
- **THEN** the system SHALL log all metrics to Weights & Biases
- **AND** the system SHALL log model hyperparameters
- **AND** the system SHALL optionally log model histograms

### Requirement: Training Entry Point

The system SHALL provide a command-line entry point for launching PyTorch Lightning training with configurable hyperparameters.

#### Scenario: Basic training launch
- **GIVEN** a terminal with Python environment and WiFo installed
- **WHEN** running `python src/train_lightning.py --dataset DS1 --size base --epochs 200`
- **THEN** the system SHALL initialize the WiFo Lightning module with base configuration
- **AND** the system SHALL load the specified dataset
- **AND** the system SHALL train for 200 epochs with default hyperparameters
- **AND** the system SHALL save checkpoints and logs to the appropriate directory

#### Scenario: Training with custom hyperparameters
- **GIVEN** a terminal with Python environment
- **WHEN** running `python src/train_lightning.py --lr 1e-3 --batch_size 256 --mask_ratio 0.75`
- **THEN** the system SHALL override default hyperparameters with specified values
- **AND** the system SHALL log the hyperparameter configuration to TensorBoard

### Requirement: Unit Test Coverage

The system SHALL provide comprehensive unit tests for all PyTorch Lightning components.

#### Scenario: LightningModule tests
- **GIVEN** the test suite in `test/test_lightning_module.py`
- **WHEN** running the tests
- **THEN** the system SHALL verify module initialization with different model sizes
- **AND** the system SHALL verify forward pass with each masking strategy
- **AND** the system SHALL verify loss computation
- **AND** the system SHALL verify optimizer configuration
- **AND** the system SHALL verify learning rate scheduling

#### Scenario: DataModule tests
- **GIVEN** the test suite in `test/test_data_module.py`
- **WHEN** running the tests
- **THEN** the system SHALL verify data loading from .mat files
- **AND** the system SHALL verify train/val/test splitting
- **AND** the system SHALL verify data normalization
- **AND** the system SHALL verify complex-to-real tensor conversion
- **AND** the system SHALL verify DataLoader creation

#### Scenario: Integration tests
- **GIVEN** the test suite in `test/test_training_integration.py`
- **WHEN** running the tests
- **THEN** the system SHALL verify a complete training step
- **AND** the system SHALL verify validation loop
- **AND** the system SHALL verify checkpoint saving and loading
- **AND** the system SHALL verify multi-GPU training setup (if GPUs available)
