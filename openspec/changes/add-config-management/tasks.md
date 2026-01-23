# Tasks: Add YAML Configuration Management System

## Phase 1: Core Configuration Module

- [ ] 1.1 Create `src/config/` directory structure
- [ ] 1.2 Implement `src/config/loader.py` with YAML loading
- [ ] 1.3 Implement `src/config/parser.py` with CLI override parsing
- [ ] 1.4 Implement `src/config/validator.py` with validation rules
- [ ] 1.5 Add `src/config/__init__.py` with public API
- [ ] 1.6 Write unit tests for config loading

## Phase 2: Default Configuration Files

- [ ] 2.1 Create `configs/` directory at project root
- [ ] 2.2 Create `configs/base_training.yaml` (paper baseline)
- [ ] 2.3 Create `configs/quick_test.yaml` (fast iteration)
- [ ] 2.4 Create `configs/tiny_model.yaml`
- [ ] 2.5 Create `configs/evaluation/zero_shot.yaml`
- [ ] 2.6 Add README in `configs/` explaining usage

## Phase 3: Integrate with Training Script

- [ ] 3.1 Modify `src/train.py` to accept `--config` argument
- [ ] 3.2 Replace hardcoded defaults with config loading
- [ ] 3.3 Implement CLI override parsing for training
- [ ] 3.4 Add config validation before training starts
- [ ] 3.5 Save config to checkpoint directory
- [ ] 3.6 Log config to TensorBoard
- [ ] 3.7 Test training with config file only
- [ ] 3.8 Test training with config + CLI overrides
- [ ] 3.9 Test backward compatibility (no config file)

## Phase 4: Integrate with Evaluation Script

- [ ] 4.1 Modify `src/eval.py` to accept `--config` argument
- [ ] 4.2 Replace hardcoded defaults with config loading
- [ ] 4.3 Implement CLI override parsing for evaluation
- [ ] 4.4 Add config validation before evaluation starts
- [ ] 4.5 Save config to results directory
- [ ] 4.6 Test evaluation with config file only
- [ ] 4.7 Test evaluation with config + CLI overrides

## Phase 5: Documentation and Examples

- [ ] 5.1 Update `src/README.md` with config usage
- [ ] 5.2 Create `docs/config-guide.md` with detailed examples
- [ ] 5.3 Add config examples to training documentation
- [ ] 5.4 Document all available config parameters
- [ ] 5.5 Add troubleshooting section for common config issues

## Dependencies

- Phase 2 depends on Phase 1 (need config module first)
- Phase 3 depends on Phase 1 and 2 (need config module and default configs)
- Phase 4 depends on Phase 1 and 2 (need config module and default configs)
- Phase 5 can be done incrementally alongside other phases

## Parallelizable Work

- Phases 1 and 2 can be developed in parallel (config module and config files are independent)
- Phases 3 and 4 can be done in parallel (train and eval scripts are independent)
- Phase 5 tasks can be done alongside implementation
