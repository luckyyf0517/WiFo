# WiFo Source Code Structure

## Directory Layout

```
src/
├── data/           # Data loading and preprocessing
├── models/         # WiFo model architecture
├── training/       # PyTorch Lightning module wrapper
├── utils/          # Utility functions
└── train.py        # Main training entry point
```

## Training Usage

```bash
python src/train.py --dataset DS1 --size base --epochs 200
```

## Legacy Code

Old training scripts have been moved to `tmp/reference/`:
- `train.py` - Old TrainLoop class (manual PyTorch training)
- `main.py` - Old entry point

## Module Import Examples

```python
# Import model
from src.models.model import WiFo_model

# Import Lightning module
from src.training.lightning_module import WiFoLightningModule

# Import DataModule
from src.data.data_module import WiFoDataModule
```
