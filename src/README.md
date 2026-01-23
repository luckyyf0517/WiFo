# WiFo Source Code Structure

## Directory Layout

```
src/
├── data/           # Data loading and preprocessing
├── models/         # WiFo model architecture
├── training/       # PyTorch Lightning module wrapper
├── utils/          # Utility functions
├── train.py        # Main training entry point
└── eval.py         # Evaluation/Inference entry point
```

## Training Usage

```bash
python src/train.py --dataset DS1 --size base --epochs 200
```

## Evaluation Usage

```bash
python src/eval.py --dataset "D1*D2*D3" --size base --checkpoint_path weights/lightning/wifo_base.ckpt
```

## Weights

Pre-trained weights are available in `weights/lightning/`:
- `wifo_tiny.ckpt` - Tiny model (366K params)
- `wifo_little.ckpt` - Little model (1.4M params)
- `wifo_small.ckpt` - Small model (5.4M params)
- `wifo_base.ckpt` - Base model (21.5M params)

To convert legacy `.pkl` weights to Lightning format:
```bash
python scripts/convert_weights.py --size base --input weights/release/wifo_base.pkl --t_patch_size 4 --patch_size 4
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
