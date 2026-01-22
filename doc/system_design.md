# 🏗️ WiFo: Wireless Foundation Model Architecture & Logic

## 1. System Overview

**WiFo** is a Masked Autoencoder (MAE) based architecture designed for **Space-Time-Frequency (STF)** channel reconstruction. It treats channel prediction (both time and frequency domains) as a unified "image" reconstruction problem.

*   **Input:** Complex Channel State Information (CSI).
*   **Core Mechanism:** 3D Patching → Masking → Transformer Encoder → Transformer Decoder → Reconstruction.
*   **Key Innovation:** **STF-PE** (Space-Time-Frequency Positional Encoding) to handle 3D physics-compliant coordinates.

### Code Structure

The implementation is organized as follows:

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| [main.py](src/main.py) | Entry point & configuration | `main()`, `create_argparser()` |
| [model.py](src/model.py) | MAE architecture | `WiFo`, `WiFo_model()`, `Attention`, `Block` |
| [embed.py](src/embed.py) | Token & positional embeddings | `TokenEmbedding`, `DataEmbedding`, `get_1d_sincos_pos_embed_from_grid_with_resolution` |
| [mask_strategy.py](src/mask_strategy.py) | Masking strategies | `random_masking()`, `causal_masking()`, `fre_masking()` |
| [train.py](src/train.py) | Training loop | `TrainLoop` |
| [dataloader.py](src/dataloader.py) | Data loading | `data_load_main()`, `MyDataset` |

---

## 2. Network Architecture

### 2.1 Model Size Configurations

The model supports four sizes via [WiFo_model()](src/model.py#L19-L92):

| Size | embed_dim | depth | decoder_embed_dim | decoder_depth | num_heads |
|------|-----------|-------|------------------|---------------|-----------|
| **tiny** | 64 | 6 | 64 | 4 | 8 |
| **little** | 128 | 6 | 128 | 4 | 8 |
| **small** | 256 | 6 | 256 | 4 | 8 |
| **base** | 512 | 6 | 512 | 4 | 8 |

```python
# src/model.py:21-37
if args.size == 'small':
    model = WiFo(
        embed_dim=256,
        depth=6,
        decoder_embed_dim=256,
        decoder_depth=4,
        num_heads=8,
        decoder_num_heads=8,
        mlp_ratio=2,
        t_patch_size=args.t_patch_size,
        patch_size=args.patch_size,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pos_emb=args.pos_emb,
        no_qkv_bias=bool(args.no_qkv_bias),
        args=args,
    )
```

### 2.2 Input Data Representation & Pre-processing

The input is a complex tensor $H \in \mathbb{C}^{T \times K \times N}$, where:
*   $T$: Time steps (snapshots).
*   $K$: Frequency sub-carriers.
*   $N$: Number of antennas.

**Data Loading** ([dataloader.py:30-46](src/dataloader.py#L30-L46)):

```python
# Load complex CSI data from .mat file
X_test = hdf5storage.loadmat(folder_path_test)
X_test_complex = torch.tensor(np.array(X_test['X_val'], dtype=complex)).unsqueeze(1)

# Convert to real + imaginary parts
X_test = torch.cat((X_test_complex.real, X_test_complex.imag), dim=1).float()
```

### 2.3 CSI Embedding (3D Patching)

**TokenEmbedding** ([embed.py:11-27](src/embed.py#L11-L27)) uses 3D convolution to convert the input tensor to patches:

```python
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, t_patch_size, patch_size):
        super(TokenEmbedding, self).__init__()
        kernel_size = [t_patch_size, patch_size, patch_size]
        self.tokenConv = nn.Conv3d(
            in_channels=c_in,      # 2 (real + imaginary)
            out_channels=d_model,   # embed_dim
            kernel_size=kernel_size,
            stride=kernel_size      # Non-overlapping patches
        )
```

*   **Patch Size:** $(t, k, n)$ configurable via `t_patch_size` and `patch_size`
*   **Total Tokens ($L$):** $L = \frac{T}{t} \times \frac{K}{k} \times \frac{N}{n}$

**Forward pass** ([embed.py:21-27](src/embed.py#L21-L27)):

```python
def forward(self, x):
    x = self.tokenConv(x)           # Conv3D: [N, C, T, H, W] -> [N, D, T, H, W]
    x = x.flatten(3)                # [N, D, T, H*W]
    x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, D]
    x = x.reshape(x.shape[0], -1, x.shape[-1])  # [N, T*H*W, D]
    return x
```

### 2.4 Masking Strategy

The tokens are split into **Visible** and **Masked** sets based on a masking strategy.

**Implementation** in [model.py:557-577](src/model.py#L557-L577):

```python
def forward_encoder(self, x, mask_ratio, mask_strategy, seed=None, data=None, mode='backward', scale=None):
    N, _, T, H, W = x.shape
    x = self.Embedding(x)  # TokenEmbedding
    _, L, C = x.shape

    T = T // self.args.t_patch_size
    H = H // self.patch_size
    W = W // self.patch_size

    if mask_strategy == 'random':
        x, mask, ids_restore, ids_keep = random_masking(x, mask_ratio)
    elif mask_strategy == 'temporal':
        x, mask, ids_restore, ids_keep = causal_masking(x, mask_ratio, T=T)
    elif mask_strategy == 'fre':
        x, mask, ids_restore, ids_keep = fre_masking(x, mask_ratio, T=T, H=H, W=W)
    # ...
```

### 2.5 STF Positional Encoding (STF-PE)

**3D Sinusoidal Position Embedding** ([embed.py:195-214](src/embed.py#L195-L214)):

```python
def get_1d_sincos_pos_embed_from_grid_with_resolution(embed_dim, pos, res):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    res: resolution scaling factor
    out: (M, D)
    """
    pos = pos * res  # Apply resolution scaling
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
```

**Position embedding for encoder** ([model.py:425-472](src/model.py#L425-L472)):

The encoder supports 3D position embeddings with dimension splitting:

```python
# Dimension split for embed_dim = 256 (example)
if ED == 256:
    ED1 = 86  # temporal dimension
    ED2 = 86  # frequency dimension
    ED3 = 84  # spatial dimension

# Create 3D grid coordinates
tt, hh, ww = torch.meshgrid(t, h, w, indexing='ij')

# Generate embeddings for each dimension
emb_t = get_1d_sincos_pos_embed_from_grid_with_resolution(ED1, tt.flatten(), scale[0])
emb_h = get_1d_sincos_pos_embed_from_grid_with_resolution(ED2, hh.flatten(), scale[1])
emb_w = get_1d_sincos_pos_embed_from_grid_with_resolution(ED3, ww.flatten(), scale[2])

# Concatenate to form 3D positional embedding
pos_embed = np.concatenate([emb_t, emb_h, emb_w], axis=1)
```

### 2.6 Encoder

**Transformer Block** ([model.py:151-193](src/model.py#L151-L193)):

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, ...):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(dim, num_heads, qkv_bias=qkv_bias, ...)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, ...)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

**Multi-Head Attention** ([model.py:95-148](src/model.py#L95-L148)):

```python
class Attention(nn.Module):
    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
```

### 2.7 Decoder

**Decoder forward pass** ([model.py:600-636](src/model.py#L600-L636)):

```python
def forward_decoder(self, x, ids_restore, mask_strategy, input_size=None, data=None, scale=None):
    N = x.shape[0]
    T, H, W = input_size

    # Project to decoder dimension
    x = self.decoder_embed(x)
    C = x.shape[-1]

    # Restore full sequence with mask tokens
    if mask_strategy == 'random':
        x = random_restore(x, ids_restore, N, T, H, W, C, self.mask_token)
    elif mask_strategy == 'temporal':
        x = causal_restore(x, ids_restore, N, T, H, W, C, self.mask_token)
    elif mask_strategy == 'fre':
        x = fre_restore(x, ids_restore, N, T, H, W, C, self.mask_token)

    # Add decoder positional embedding
    decoder_pos_embed = self.pos_embed_dec(ids_restore, N, input_size)
    x_attn = x + decoder_pos_embed

    # Apply decoder blocks
    for blk in self.decoder_blocks:
        x_attn = blk(x_attn)
    x_attn = self.decoder_norm(x_attn)

    return x_attn
```

**Mask token restoration** ([mask_strategy.py:105-115](src/mask_strategy.py#L105-L115)):

```python
def random_restore(x, ids_restore, N, T, H, W, C, mask_token):
    # Create mask tokens for masked positions
    mask_tokens = mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
    x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)
    x_ = x_.view([N, T * H * W, C])
    # Unshuffle to restore original order
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))
    return x_.view([N, T * H * W, C])
```

### 2.8 Loss Computation

**MSE Loss on masked patches** ([model.py:639-656](src/model.py#L639-L656)):

```python
def forward_loss(self, imgs, pred, mask):
    """
    imgs: [N, 2, T, H, W]
    pred: [N, t*h*w, u*p*p*2]
    mask: [N, L], 0 is keep, 1 is remove
    """
    target = self.patchify(imgs)  # Convert to patches
    assert pred.shape == target.shape

    loss = torch.abs(pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    mask = mask.view(loss.shape)

    # Loss on masked patches only
    loss1 = (loss * mask).sum() / mask.sum()
    # Loss on visible patches (for monitoring)
    loss2 = (loss * (1-mask)).sum() / (1-mask).sum()

    return loss1, loss2, target
```

---

## 3. Self-Supervised Pre-training

### 3.1 Training Loop

**TrainLoop class** ([train.py:12-151](src/train.py#L12-L151)):

```python
class TrainLoop:
    def __init__(self, args, writer, model, test_data, device, early_stop=5):
        self.args = args
        self.model = model
        self.test_data = test_data
        self.device = device
        self.opt = AdamW([p for p in self.model.parameters() if p.requires_grad],
                        lr=args.lr, weight_decay=args.weight_decay)

        # Mask ratios for different strategies
        self.mask_list = {'random': [0.85], 'temporal': [0.5], 'fre': [0.5]}
```

**Mask strategy selection** ([train.py:117-125](src/train.py#L117-L125)):

```python
def mask_select(self, name):
    if self.args.mask_strategy_random == 'none':
        mask_strategy = self.args.mask_strategy
        mask_ratio = self.args.mask_ratio
    else:
        # Randomly choose among three strategies
        mask_strategy = random.choice(['random', 'temporal', 'fre'])
        mask_ratio = random.choice(self.mask_list[mask_strategy])
    return mask_strategy, mask_ratio
```

### 3.2 Three Masking Strategies

#### 3.2.1 Random Masking

**Implementation** ([mask_strategy.py:5-33](src/mask_strategy.py#L5-L33)):

```python
def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    # Generate random noise for shuffling
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)  # Ascending: small=keep, large=remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # Inverse shuffle

    # Keep first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # Generate binary mask: 0=keep, 1=remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore, ids_keep
```

#### 3.2.2 Temporal (Causal) Masking

**Implementation** ([mask_strategy.py:35-67](src/mask_strategy.py#L35-L67)):

```python
def causal_masking(x, mask_ratio, T):
    N, L, D = x.shape
    x = x.reshape(N, T, L//T, D)
    N, T, L, C = x.shape

    len_keep = int(T * (1 - mask_ratio))

    # Use ordered indices for temporal masking
    noise = torch.arange(T).unsqueeze(dim=0).repeat(N, 1)
    noise = noise.to(x)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep first temporal subset (causal)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(2).unsqueeze(-1).repeat(1, 1, L, D))

    # Generate binary mask
    mask = torch.ones([N, T, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore.unsqueeze(2).repeat(1, 1, L)).reshape(N, -1)

    return x_masked, mask, ids_restore, ids_keep
```

#### 3.2.3 Frequency Masking

**Implementation** ([mask_strategy.py:69-100](src/mask_strategy.py#L69-L100)):

```python
def fre_masking(x, mask_ratio, T, H, W):
    N, L, D = x.shape
    x = x.reshape(N, T, H, W, D)

    len_keep = int(W * (1 - mask_ratio))

    # Use ordered indices along frequency dimension
    noise = torch.arange(W).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).repeat(N, T, H, 1)
    noise = noise.to(x)

    ids_shuffle = torch.argsort(noise, dim=3)
    ids_restore = torch.argsort(ids_shuffle, dim=3)

    # Keep first frequency subset
    ids_keep = ids_shuffle[:, :, :, :len_keep]
    x_masked = torch.gather(x, dim=3, index=ids_keep.unsqueeze(4).repeat(1, 1, 1, 1, D))

    # Generate binary mask
    mask = torch.ones([N, T, H, W], device=x.device)
    mask[:, :, :, :len_keep] = 0
    mask = torch.gather(mask, dim=3, index=ids_restore).reshape(N, -1)

    return x_masked, mask, ids_restore, ids_keep
```

### 3.3 Evaluation

**NMSE computation** ([train.py:33-56](src/train.py#L33-L56)):

```python
def Sample(self, test_data, step, mask_ratio, mask_strategy, seed=None, dataset='', index=0):
    with torch.no_grad():
        error_nmse = 0
        num = 0

        for _, batch in enumerate(test_data[index]):
            loss, _, pred, target, mask = self.model_forward(batch, self.model, mask_ratio,
                                                             mask_strategy, seed=seed, data=dataset, mode='forward')

            dim1 = pred.shape[0]
            pred_mask = pred.squeeze(dim=2)
            target_mask = target.squeeze(dim=2)

            # Extract masked predictions
            y_pred = pred_mask[mask==1].reshape(-1, 1).reshape(dim1, -1).detach().cpu().numpy()
            y_target = target_mask[mask==1].reshape(-1, 1).reshape(dim1, -1).detach().cpu().numpy()

            # Compute NMSE
            error_nmse += np.sum(np.mean(np.abs(y_target - y_pred) ** 2, axis=1) /
                                 np.mean(np.abs(y_target) ** 2, axis=1))
            num += y_pred.shape[0]

    nmse = error_nmse / num
    return nmse
```

---

## 4. Inference / Downstream Tasks

### 4.1 Forward Pass

**Main forward function** ([model.py:659-686](src/model.py#L659-L686)):

```python
def forward(self, imgs, mask_ratio=0.5, mask_strategy='random', seed=None, data='none'):
    imgs = torch.stack(imgs).squeeze(1)

    # Add noise (SNR = 20dB)
    snr_db = 20
    noise_power = torch.mean(imgs ** 2) * 10 ** (-snr_db / 10)
    noise = torch.randn_like(imgs) * torch.sqrt(noise_power)
    imgs_n = imgs + noise

    scale = [1, 1, 1]
    T, H, W = imgs_n.shape[2:]

    # Encoder
    latent, mask, ids_restore, input_size = self.forward_encoder(imgs_n, mask_ratio, mask_strategy,
                                                                   seed=seed, data=data, mode='backward', scale=scale)

    # Decoder
    pred = self.forward_decoder(latent, ids_restore, mask_strategy, input_size=input_size, data=data, scale=scale)

    # Predictor projection
    pred = self.decoder_pred(pred)

    # Convert to complex
    Len = self.t_patch_size * self.patch_size ** 2
    pred_complex = pred[:, :, :Len] + 1j * pred[:, :, Len:]

    # Loss computation
    loss1, loss2, target = self.forward_loss(imgs_n, pred_complex, mask)

    return loss1, loss2, pred_complex, target, mask
```

### 4.2 Time-Domain Channel Prediction

**Goal:** Predict future $T - T_h$ steps given history $T_h$.

1.  **Input Construction (Zero-Padding):** Create input tensor where future is zeros
2.  **Forced Masking:** Apply **temporal masking** strategy (mask tokens where $t > T_h$)
3.  **Reconstruction:** Model outputs $H_{out}$
4.  **Extraction:** $H_{pred} = H_{out}[T_h+1:T, :, :]$

### 4.3 Frequency-Domain Channel Prediction

**Goal:** Predict sub-carriers $K_u+1 : K$ given $1 : K_u$.

1.  **Input Construction:** Zero-pad unknown frequency bins
2.  **Forced Masking:** Apply **frequency masking** strategy (mask tokens where $k > K_u$)
3.  **Extraction:** $H_{pred} = H_{out}[:, K_u+1:K, :]$

---

## 5. Configuration & Hyperparameters

### 5.1 Default Parameters

From [main.py:36-79](src/main.py#L36-L79):

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| **Experimental** | `dataset` | `'DS1'` | Dataset name |
| | `his_len` | `6` | History length |
| | `pred_len` | `6` | Prediction length |
| | `few_ratio` | `0.5` | Few-shot ratio |
| **Model** | `mask_ratio` | `0.5` | Masking ratio |
| | `patch_size` | `4` | Spatial patch size |
| | `t_patch_size` | `2` | Temporal patch size |
| | `size` | `'middle'` | Model size (tiny/little/small/base) |
| | `pos_emb` | `'SinCos'` | Position embedding type |
| **Training** | `lr` | `1e-3` | Learning rate |
| | `min_lr` | `1e-5` | Minimum learning rate |
| | `weight_decay` | `0.05` | Weight decay |
| | `batch_size` | `256` | Batch size |
| | `total_epoches` | `10000` | Total training epochs |
| | `clip_grad` | `0.05` | Gradient clipping |
| | `lr_anneal_steps` | `200` | LR annealing steps |

### 5.2 Model Initialization

**Entry point** ([main.py:83-123](src/main.py#L83-L123)):

```python
def main():
    th.autograd.set_detect_anomaly(True)
    args = create_argparser().parse_args()

    # Set random seed
    setup_init(100)

    # Load data
    test_data = data_load_main(args)

    # Create experiment directory
    args.folder = 'Dataset_{}_Task_{}_FewRatio_{}_{}_{}/'.format(
        args.dataset, args.task, args.few_ratio, args.size, args.note)
    args.model_path = './experiments/{}'.format(args.folder)

    # Initialize model
    model = WiFo_model(args=args).to(device)

    # Train
    TrainLoop(args=args, writer=writer, model=model, test_data=test_data,
              device=device, early_stop=args.early_stop).run_loop()
```

---

## 6. Complete Data Flow (D1 Dataset Example)

This section documents the complete data flow through the WiFo model using D1 dataset as an example.

### 6.1 Input Specifications

**D1 Dataset:**
- Shape: [N=1000, K=24, T=4, Ant=128]
- N: number of samples
- K: subcarriers (24)
- T: timesteps (4)
- Ant: antennas (128)

### 6.2 Data Loading Pipeline

**Step 1: Raw Data Loading**
```
X_test['X_val']: [1000, 24, 4, 128]  (complex128)
```

**Step 2: Real/Imaginary Split**
```python
# dataloader.py:38-42
X_test_real = X_test_complex.real.unsqueeze(1)  # [1000, 1, 24, 4, 128]
X_test_imag = X_test_complex.imag.unsqueeze(1)  # [1000, 1, 24, 4, 128]
X_test = torch.cat((X_test_real, X_test_imag), dim=1).float()
# Output: [1000, 2, 24, 4, 128]
```

**Step 3: DataLoader Batching**
```
batch: [128, 1, 2, 24, 4, 128]  # added batch dimension
batch.squeeze(1): [128, 2, 24, 4, 128]
```

### 6.3 Tokenization (Conv3D)

**Input:** [N, 2, K=24, T=4, Ant=128]

**Conv3D Configuration:**
```python
# embed.py:14-16
kernel_size = [t_patch_size=4, patch_size=4, patch_size=4]
stride = [4, 4, 4]  # non-overlapping patches
in_channels = 2  # real + imaginary
out_channels = 256  # embedding dimension
```

**After Conv3D:**
```
[N, 256, T=4/4=1, K=24/4=6, Ant=128/4=32]
= [128, 256, 1, 6, 32]
```

**Flatten & Reshape:**
```python
# embed.py:24-26
x.flatten(3):              [128, 256, 1, 192]
torch.einsum("ncts->ntsc"): [128, 1, 192, 256]
reshape:                   [128, 192, 256]
```

**Token Count:** 1 x 6 x 32 = 192 tokens, each 256-dimensional

### 6.4 Masking Stage (Temporal Masking, ratio=0.5)

**Input:** [N, 192, 256]

**Temporal Masking Process:**
```python
# mask_strategy.py:35-67
x = x.reshape(N, T=1, HxW=192, 256)
len_keep = int(T * (1 - mask_ratio)) = int(1 * 0.5) = 0

# For T=1 with mask_ratio=0.5, special handling preserves all tokens
```

**Outputs:**
- x_masked: [N, 96, 256] - kept tokens (50%)
- mask: [N, 192] - binary mask (0=keep, 1=remove)
- ids_restore: [N, 192] - indices to restore full sequence
- ids_keep: [N, 96] - indices of kept tokens

### 6.5 Position Encoding (SinCos_3D)

**Dimension Split for embed_dim=256:**
```python
# model.py:438-441
ED1 = 86  # temporal dimension
ED2 = 86  # frequency (subcarrier) dimension
ED3 = 84  # spatial (antenna) dimension
```

**3D Grid Generation:**
```python
T, H, W = 1, 6, 32
tt, hh, ww = torch.meshgrid(t, h, w, indexing='ij')
# tt: [1, 6, 32], hh: [1, 6, 32], ww: [1, 6, 32]
```

**Position Embedding Calculation:**
```python
# embed.py:195-214
emb_t = get_1d_sincos_pos_embed_from_grid_with_resolution(86, tt.flatten(), scale[0])
emb_h = get_1d_sincos_pos_embed_from_grid_with_resolution(86, hh.flatten(), scale[1])
emb_w = get_1d_sincos_pos_embed_from_grid_with_resolution(84, ww.flatten(), scale[2])

pos_embed = np.concatenate([emb_t, emb_h, emb_w], axis=1)
# pos_embed: [192, 256]
```

**Add to Kept Tokens:**
```python
# model.py:586-588
pos_embed_sort = torch.gather(pos_embed, dim=1, index=ids_keep...)
# pos_embed_sort: [N, 96, 256]

x_attn = x_masked + pos_embed_sort: [N, 96, 256]
```

### 6.6 Encoder Transformer

**Input:** [N, 96, 256] - 96 visible tokens

**Transformer Block (6 layers):**
```python
# model.py:151-193
for blk in self.blocks:  # 6 blocks
    x = Block(x)
    # Block structure:
    #   LayerNorm -> Multi-Head Attention (8 heads) -> Residual
    #   LayerNorm -> MLP (256 -> 512 -> 256) -> Residual

# Multi-Head Attention:
#   Q, K, V: [N, 8, 96, 32]  (d_model/num_heads = 256/8 = 32)
#   Attention: softmax(QK^T / sqrt(d)) @ V
#   Output: [N, 96, 256]
```

**Output:** [N, 96, 256] - encoded visible tokens

### 6.7 Decoder Embedding & Restore

**Decoder Projection:**
```python
# model.py:608
x = self.decoder_embed(x)  # Linear(256 -> 256)
# x: [N, 96, 256]
```

**Restore Full Sequence:**
```python
# mask_strategy.py:105-115 (temporal_restore)
mask_tokens = mask_token.repeat(N, 192 - 96, 1)  # [N, 96, 256]
x_ = torch.cat([x, mask_tokens], dim=1)  # [N, 192, 256]

# Unshuffle to restore original order
x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 256))
# x_: [N, 192, 256]
```

**Result:** [N, 192, 256] - full sequence with mask tokens at masked positions

### 6.8 Decoder Position Encoding

**Add Decoder Position Embedding:**
```python
# model.py:624-626
decoder_pos_embed = self.pos_embed_dec(ids_restore, N, input_size)
# decoder_pos_embed: [N, 192, 256]

x_attn = x_ + decoder_pos_embed: [N, 192, 256]
```

### 6.9 Decoder Transformer

**Input:** [N, 192, 256] - full sequence including mask tokens

**Transformer Blocks (4 layers):**
```python
# model.py:635-637
for blk in self.decoder_blocks:  # 4 blocks
    x_attn = Block(x_attn)
    # Same block structure as encoder
```

**Self-Attention Mechanism:**
- Visible tokens (encoded features) attend to all tokens
- Information propagates from visible to masked positions
- Mask tokens learn to predict missing content

**Output:** [N, 192, 256] - all tokens reconstructed

### 6.10 Prediction Projection

**Project to Patch Content:**
```python
# model.py:679
pred = self.decoder_pred(x_attn)  # Linear(256 -> patch_content)
```

**Patch Content Size:**
```
t_patch × patch² × 2 = 4 × 4 × 4 × 2 = 128  (real+imag)
or for complex: t_patch × patch² = 4 × 4 × 4 = 64
```

**Output:**
```
pred: [N, 192, 64]  (complex predictions)
pred_complex = pred[:, :, :64] + 1j * pred[:, :, 64:]
```

### 6.11 Loss Computation

**Patchify Target:**
```python
# model.py:649
target = self.patchify(imgs_n)
# imgs_n: [N, 2, 4, 24, 128]
# target: [N, 192, 64]  (complex)
```

**MSE Loss on Masked Patches:**
```python
# model.py:654-658
loss = torch.abs(pred - target) ** 2  # [N, 192]
loss = loss.mean(dim=-1)  # [N, 192]

mask: [N, 192]  # 0=keep, 1=remove

# Loss only on masked patches
loss1 = (loss * mask).sum() / mask.sum()  # primary loss
loss2 = (loss * (1-mask)).sum() / (1-mask).sum()  # monitoring loss
```

### 6.12 Token Count for All Datasets

| Dataset | Shape (K,T,Ant) | Tokens | Calculation |
|---------|----------------|--------|-------------|
| D1 | (24,4,128) | 192 | (4/4)×(24/4)×(128/4) |
| D2 | (24,8,128) | 384 | (8/4)×(24/4)×(128/4) |
| D3 | (16,8,64) | 128 | (8/4)×(16/4)×(64/4) |
| D4 | (16,32,32) | 256 | (32/4)×(16/4)×(32/4) |
| D5 | (24,4,64) | 96 | (4/4)×(24/4)×(64/4) |
| D6 | (24,8,128) | 384 | (8/4)×(24/4)×(128/4) |
| D7 | (16,32,32) | 256 | (32/4)×(16/4)×(32/4) |
| D8 | (16,16,64) | 256 | (16/4)×(16/4)×(64/4) |
| D9 | (24,4,128) | 192 | (4/4)×(24/4)×(128/4) |
| D10 | (24,8,64) | 192 | (8/4)×(24/4)×(64/4) |
| D11 | (16,16,64) | 256 | (16/4)×(16/4)×(64/4) |
| D12 | (16,32,32) | 256 | (32/4)×(16/4)×(32/4) |
| D13 | (24,16,64) | 384 | (16/4)×(24/4)×(64/4) |
| D14 | (24,8,128) | 384 | (8/4)×(24/4)×(128/4) |
| D15 | (16,16,64) | 256 | (16/4)×(16/4)×(64/4) |
| D16 | (16,32,32) | 256 | (32/4)×(16/4)×(32/4) |

### 6.13 Summary Table

| Stage | Input Shape | Output Shape | Key Operation |
|-------|-------------|--------------|---------------|
| Data Loading | [1000,24,4,128] | [128,2,24,4,128] | Real/imag split + batch |
| Tokenization | [128,2,24,4,128] | [128,192,256] | Conv3D + flatten |
| Masking | [128,192,256] | [128,96,256] | Temporal masking (50%) |
| Pos Encoding | [128,96,256] | [128,96,256] | Add 3D sinusoidal PE |
| Encoder | [128,96,256] | [128,96,256] | 6× Transformer Block |
| Decoder Restore | [128,96,256] | [128,192,256] | Insert mask tokens |
| Decoder Pos Enc | [128,192,256] | [128,192,256] | Add decoder PE |
| Decoder | [128,192,256] | [128,192,256] | 4× Transformer Block |
| Prediction | [128,192,256] | [128,192,64] | Linear projection |
| Loss | pred vs target | scalar | MSE on masked |

---

## 7. Key Implementation Details

### 6.1 Complex Number Handling

The model processes complex-valued CSI data by:
1. Stacking real and imaginary parts as separate channels ([dataloader.py:38](src/dataloader.py#L38))
2. Processing through the network as real-valued tensors
3. Reconstructing complex output from real/imaginary predictions ([model.py:679-680](src/model.py#L679-L680)):

```python
pred_complex = pred[:, :, :Len] + 1j * pred[:, :, Len:]
```

### 6.2 Noise Injection

During training, Gaussian noise is added to simulate real-world conditions ([model.py:663-666](src/model.py#L663-L666)):

```python
snr_db = 20
noise_power = torch.mean(imgs ** 2) * 10 ** (-snr_db / 10)
noise = torch.randn_like(imgs) * torch.sqrt(noise_power)
imgs_n = imgs + noise
```

### 6.3 Position Embedding Modes

The model supports multiple position embedding modes ([model.py:582-591](src/model.py#L582-L591)):

*   **`'trivial'`**: Learnable embeddings for spatial and temporal dimensions
*   **`'SinCos'`**: Fixed sinusoidal embeddings (2D spatial + 1D temporal)
*   **`'SinCos_3D'`**: Full 3D sinusoidal embeddings with resolution scaling
*   **`'None'`**: No positional encoding

---

## 8. Summary

WiFo implements a sophisticated Masked Autoencoder specifically designed for wireless CSI reconstruction:

1. **Modular Architecture**: Clean separation between model, training, data loading, embeddings, and masking strategies
2. **Multiple Masking Strategies**: Random, temporal, and frequency masking for robust feature learning
3. **Complex Data Support**: Proper handling of complex-valued CSI data throughout the pipeline
4. **Flexible Position Encoding**: Multiple PE modes including physics-compliant 3D sinusoidal embeddings
5. **Zero-Shot Inference**: Prediction tasks formulated as masked reconstruction problems
