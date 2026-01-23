# Change: Add YAML Configuration Management System

## Why

The current training and evaluation scripts use hardcoded argparse defaults with 50+ parameters, making it difficult to:
- Manage different experiment configurations (e.g., paper baselines vs. custom experiments)
- Reproduce experiments without complex command-line arguments
- Share configurations between team members
- Maintain version-controlled configuration files
- Override specific parameters without reproducing the entire command

A YAML-based configuration system with command-line override support will provide:
- **Reproducibility**: Configurations stored in version-controlled YAML files
- **Flexibility**: Override any parameter via `--config.path.nested.param value`
- **Simplicity**: Default configurations defined once, referenced everywhere
- **Organization**: Logical grouping of related parameters
- **Consistency**: All parameters use the same dot notation, no shortcuts

## What Changes

- **ADDED**: YAML configuration file structure with default configurations
- **ADDED**: Configuration loading module with YAML parsing and CLI override
- **ADDED**: Pre-configured YAML files for common scenarios (paper baselines, quick test, etc.)
- **MODIFIED**: `src/train.py` to use `--config` argument with parameter overrides
- **MODIFIED**: `src/eval.py` to use `--config` argument with parameter overrides
- **ADDED**: Configuration validation to ensure required parameters are present
- **ADDED**: Helper utilities for nested parameter access and override

## Impact

- **Affected code**:
  - `src/train.py` - Modified to accept `--config` argument
  - `src/eval.py` - Modified to accept `--config` argument
  - `src/config/` - NEW: Configuration management module
  - `configs/` - NEW: YAML configuration files directory

- **Backward compatibility**: When no `--config` is specified, the original argparse behavior is preserved

- **User experience**:
  ```bash
  # Use config file
  python src/train.py --config configs/base_training.yaml

  # Override specific parameters (dot notation only)
  python src/train.py --config configs/base.yaml --data.dataset DS2

  # Override multiple parameters
  python src/train.py --config configs/base.yaml --model.size tiny --training.batch_size 64

  # Deep nested override
  python src/train.py --config configs/base.yaml --training.mask.ratio 0.75

  # No config - backward compatible
  python src/train.py --data.dataset DS1 --model.size base --training.batch_size 128
  ```

**Important**: All parameter overrides MUST use dot notation matching the YAML structure. No shortcut arguments like `--dataset` or `--size` are provided. This ensures consistency and clarity.

## Alternatives Considered

1. **JSON configs**: Less readable for humans, no comments support
2. **Python configs**: Too flexible, harder to validate, not as portable
3. **Hydra/OmegaConf**: Powerful but heavy dependency; chose lightweight custom solution
4. **TLDFConfig**: Simple but doesn't support nested overrides as naturally

## Open Questions

1. Should we use a library like `hydra-core` or implement a lightweight solution?
   - **Decision**: Lightweight custom solution to minimize dependencies

2. What should the default config file location be?
   - **Decision**: `configs/` directory at project root

3. Should we support config file inheritance/merging?
   - **Decision**: Not in initial implementation, but design to allow future extension
