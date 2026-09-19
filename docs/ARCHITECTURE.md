# CAMFN — Cross-Attention Multimodal Fusion Network

Architecture reference for this skeleton. Maps directly onto the "Neural
Network Core" section of the system spec. See `DATA_SOURCES.md` for what
real data each modality path is exercised against.

## 1. Data flow

```
 MRI slice (2D)         EEG window (C x T)      Drawing image (2D)     Voice features (22,)
      |                        |                        |                     |
 ImageEncoder2D           EEGEncoder              ImageEncoder2D      TabularTemporalEncoder
 (CNN -> pooled vec)   (1D-CNN+BiLSTM+TemporalAttn)  (CNN -> pooled vec)   (Transformer -> pooled vec)
      |                        |                        |                     |
      +------------------------+------------------------+---------------------+
                                        |
                         project each to shared d_model
                                        |
                     stack as modality-token sequence  [B, M_present, d_model]
                                        |
                   +--------------------------------------------+
                   |   ModalityTokenTransformer (self+cross attn |
                   |   over the modality-token sequence)         |
                   +--------------------------------------------+
                                        |
                   +--------------------------------------------+
                   |   Pairwise CrossAttentionLayer(s)            |
                   |   e.g. Q=MRI tokens, K/V=EEG tokens          |
                   |   (spatial-vs-temporal cross attention)     |
                   +--------------------------------------------+
                                        |
                   +--------------------------------------------+
                   |   DynamicGatedFusion                        |
                   |   gate_m = sigmoid(MLP([tok_m, mask]))       |
                   |   fused  = sum_m gate_m * tok_m              |
                   +--------------------------------------------+
                                        |
                        +---------------+----------------+
                        |                                |
                 DiagnosisHead                      SeverityHead(s)
             (HC/AD/PD/Epilepsy/Comorbid)      (CDR-like, UPDRS-III-like,
                                                 SeizureFreqIndex-like)
```

## 2. Modality encoders

- **`ImageEncoder2D`** (`src/models/encoders/image_encoder.py`) — a small
  4-block CNN (conv-BN-ReLU-maxpool ×4) + global average pool + linear
  projection to `d_model`. Used for both the MRI-slice modality and the
  Parkinson drawing modality (separate weights, same class). A
  `Volumetric3DEncoder` stub (3D-conv version) is included for true NIfTI
  volumes; not exercised by the datasets in this repo.

- **`EEGEncoder`** — 1D-CNN stack over the channel axis, then a
  bidirectional LSTM over time, then a temporal-attention pooling layer
  (learned query attends over LSTM outputs across time — standard additive
  attention, `softmax(w^T tanh(W h_t))`). Works for both multi-channel
  CHB-MIT windows and single-channel 178-sample Epileptic-Seizure-Recognition
  rows (channel dim = 1).

- **`TabularTemporalEncoder`** — treats a feature vector as a length-1
  "sequence" (or a true sequence, if given one) and runs it through a small
  Transformer encoder (`nn.TransformerEncoderLayer` ×2) with a learned
  `[CLS]`-style pooling token. Used for the UCI voice features.

Every encoder outputs a single `[B, d_model]` embedding per present modality,
plus (optionally) a token sequence `[B, L, d_model]` before pooling, which
`CrossAttentionLayer` can use for finer-grained fusion.

## 3. Fusion engine

### 3.1 Cross-attention (pairwise)
For modalities \(a\) (query) and \(b\) (key/value), with token sequences
\(X_a \in \mathbb{R}^{L_a \times d}\), \(X_b \in \mathbb{R}^{L_b \times d}\):

$$Q = X_a W_Q,\quad K = X_b W_K,\quad V = X_b W_V$$
$$\text{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

`CrossAttentionLayer` implements exactly this (multi-head, residual + LayerNorm
+ feed-forward), letting e.g. MRI spatial tokens query CHB-MIT EEG temporal
tokens as specified in the spec's fusion section.

### 3.2 Modality-token transformer
For the coarser, always-available fusion path, each modality's pooled
embedding becomes one token; a small Transformer encoder does self-attention
across the (present) modality tokens, letting every modality attend to every
other one regardless of how many are missing this batch.

### 3.3 Dynamic Gated Fusion Unit
Given per-modality tokens \(t_m\) and a binary presence mask \(\mu_m\)
(1 = modality present this sample, 0 = missing/dropped):

$$g_m = \sigma\big(\mathrm{MLP}([t_m \Vert \mu_m])\big), \qquad
\hat t_m = \mu_m \cdot g_m \cdot t_m$$
$$\text{fused} = \sum_m \hat t_m \Big/ \Big(\sum_m \mu_m g_m + \epsilon\Big)$$

Missing modalities are represented by a learned placeholder embedding (not
zeros) so the gate has something principled to downweight, and the
normalization keeps the fused vector's scale stable regardless of how many
modalities are present.

## 4. Classification / severity heads
- `DiagnosisHead`: `Linear(d_model -> 5)` over `{HC, AD, PD, Epilepsy,
  Comorbid}`, trained with cross-entropy.
- `SeverityHead`: three small MLP regressors (CDR-like, UPDRS-III-like,
  Seizure-Frequency-Index-like), each trained with MSE, masked out for
  samples where that severity label doesn't apply (e.g. UPDRS only makes
  sense for PD-cohort samples) — see `src/training/losses.py::MultiTaskLoss`.

## 5. Missing/noisy modality handling
`MultimodalNeuroDataset.collate_fn` emits a presence mask alongside the
batch. `CAMFN.forward` substitutes a learned placeholder token for any
missing modality before fusion, and `DynamicGatedFusion` uses the mask to
zero that modality's contribution to the fused vector (rather than letting
the placeholder leak signal in). This is what makes the network robust to a
patient missing, say, EEG but having MRI + voice.
