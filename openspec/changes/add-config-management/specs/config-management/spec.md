# Spec: Configuration Management

## ADDED Requirements

### Requirement: YAML Configuration File Loading

The system SHALL load training and evaluation configurations from YAML files.

#### Scenario: Load config from file
- **GIVEN** a YAML configuration file at `configs/base_training.yaml`
- **WHEN** user runs `python src/train.py --config configs/base_training.yaml`
- **THEN** the system SHALL load all parameters from the YAML file
- **AND** the system SHALL parse nested structures correctly
- **AND** the system SHALL preserve data types (int, float, bool, list, str)

#### Scenario: Load default config when none specified
- **GIVEN** no `--config` argument is provided
- **WHEN** user runs `python src/train.py --dataset DS1`
- **THEN** the system SHALL use hardcoded default values
- **AND** the system SHALL maintain backward compatibility

#### Scenario: Handle missing config file
- **GIVEN** a non-existent config file path
- **WHEN** user runs `python src/train.py --config configs/missing.yaml`
- **THEN** the system SHALL exit with error code 1
- **AND** the system SHALL display a clear error message
- **AND** the system SHALL suggest available config files

### Requirement: CLI Parameter Override

The system SHALL allow overriding any configuration parameter via command-line arguments using dot notation.

#### Scenario: Override nested parameter
- **GIVEN** a config file with `training.optimizer.lr: 5e-4`
- **WHEN** user runs `python src/train.py --config base.yaml --training.optimizer.lr 1e-3`
- **THEN** the system SHALL override `training.optimizer.lr` to `1e-3`
- **AND** the system SHALL preserve all other config values

#### Scenario: Override multiple parameters
- **GIVEN** a config file with default values
- **WHEN** user runs `python src/train.py --config base.yaml --model.size tiny --training.batch_size 64`
- **THEN** the system SHALL override both `model.size` and `training.batch_size`
- **AND** the system SHALL merge all overrides correctly

#### Scenario: Override non-existent parameter
- **GIVEN** a config file
- **WHEN** user runs `python src/train.py --config base.yaml --invalid.param value`
- **THEN** the system SHALL warn about the invalid parameter
- **AND** the system SHALL continue with other valid parameters

#### Scenario: Boolean parameter override
- **GIVEN** a config file with `training.enable_checkpoint: true`
- **WHEN** user runs `python src/train.py --config base.yaml --training.enable_checkpoint false`
- **THEN** the system SHALL override to boolean `false`
- **AND** the system SHALL correctly parse the boolean value

### Requirement: Configuration Validation

The system SHALL validate configuration parameters before starting training or evaluation.

#### Scenario: Validate required parameters
- **GIVEN** a config file missing `model.size`
- **WHEN** the config is loaded
- **THEN** the system SHALL detect the missing required parameter
- **AND** the system SHALL exit with clear error listing missing parameters

#### Scenario: Validate parameter values
- **GIVEN** a config file with `model.size: invalid`
- **WHEN** the config is loaded
- **THEN** the system SHALL detect the invalid value
- **AND** the system SHALL show valid options [tiny, little, small, base]

#### Scenario: Validate parameter types
- **GIVEN** a config file with `training.batch_size: not_a_number`
- **WHEN** the config is loaded
- **THEN** the system SHALL detect the type mismatch
- **AND** the system SHALL show the expected type

### Requirement: Configuration Persistence

The system SHALL save active configuration to experiment directories for reproducibility.

#### Scenario: Save config with training run
- **GIVEN** a training run with config file and CLI overrides
- **WHEN** training starts
- **THEN** the system SHALL save the resolved config to `{output_dir}/config.yaml`
- **AND** the saved config SHALL include all CLI overrides
- **AND** the saved config SHALL match the exact configuration used

#### Scenario: Save config with evaluation run
- **GIVEN** an evaluation run with config file
- **WHEN** evaluation starts
- **THEN** the system SHALL save the resolved config to `{output_dir}/config.yaml`
- **AND** the saved config SHALL enable exact reproduction of results

### Requirement: Default Configuration Files

The system SHALL provide pre-configured YAML files for common use cases.

#### Scenario: Paper baseline configuration
- **GIVEN** the file `configs/base_training.yaml`
- **WHEN** loaded
- **THEN** it SHALL contain all hyperparameters from the paper
- **AND** it SHALL reproduce paper results when used

#### Scenario: Quick test configuration
- **GIVEN** the file `configs/quick_test.yaml`
- **WHEN** loaded
- **THEN** it SHALL use the tiny model for fast iteration
- **AND** it SHALL use minimal epochs (e.g., 10)
- **AND** it SHALL use small batch size for faster training

#### Scenario: Zero-shot evaluation configuration
- **GIVEN** the file `configs/evaluation/zero_shot.yaml`
- **WHEN** loaded
- **THEN** it SHALL configure evaluation on all test datasets
- **AND** it SHALL set `few_ratio: 0.0` for zero-shot evaluation
- **AND** it SHALL configure temporal masking strategy

### Requirement: Config Helper Utilities

The system SHALL provide helper functions for working with configurations.

#### Scenario: Flatten nested config
- **GIVEN** a nested config dict `{model: {size: "base"}}`
- **WHEN** calling `flatten_config(config)`
- **THEN** it SHALL return `{"model.size": "base"}`
- **AND** all nested keys SHALL be dot-joined

#### Scenario: Merge configs with priority
- **GIVEN** base config `{a: {x: 1}}` and override `{a: {x: 2, y: 3}}`
- **WHEN** calling `merge_configs(base, override)`
- **THEN** it SHALL return `{a: {x: 2, y: 3}}`
- **AND** override values SHALL take precedence

#### Scenario: Convert config to namespace
- **GIVEN** a config dict `{model: {size: "base"}}`
- **WHEN** calling `config_to_namespace(config)`
- **THEN** it SHALL return an object with `config.model.size == "base"`
- **AND** dot access SHALL work for all nested keys
