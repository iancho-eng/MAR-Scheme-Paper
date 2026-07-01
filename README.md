# MAR-MP: Memory-Addressed Register with Merge Planning

> A hardware-software co-design scheme for compressed address encoding in DNN sparse matrix acceleration.

[![Paper](https://img.shields.io/badge/Paper-MAR--MP-blue)](.)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](.)
[![License](https://img.shields.io/badge/License-MIT-yellow)](.)

---

## Overview

MAR-MP addresses two critical bottlenecks in deploying sparse matrix acceleration for deep neural network (DNN) inference on edge hardware:

1. **Address storage overhead** — conventional methods (CSR, MATSORT) store full-width absolute column indices for every non-zero element, which is impractical for memory-constrained edge devices.
2. **Runtime merge latency** — iterative row-merge decisions that scale poorly with matrix size introduce significant runtime branching overhead.

MAR-MP resolves both problems through two complementary techniques:

- **Delta encoding** — replaces absolute column indices with compact gap sequences between consecutive non-zero positions, reducing per-element address bit-width to 3–4 bits for typical pruned networks.
- **Offline merge planning** — precomputes the full row-merging schedule as a fixed lookup table, eliminating all runtime conditional branching during inference.

Together, these techniques reduce address metadata storage by up to **40%**, overall compressed size by up to **15%**, and encoding time by up to **24×** compared to MATSORT, with no loss of accuracy.

---

## Key Results

| Algorithm | Avg. Run Time Improvement | Avg. Size Improvement |
|-----------|--------------------------|----------------------|
| CSR       | ×1                       | ×1                   |
| HiRAC     | ×0.67                    | ×1.01                |
| MATSORT   | ×1.02                    | ×1.04                |
| **MAR-MP**| **×18.27**               | **×1.12**            |

*Evaluated at 75% sparsity across variant matrix sizes.*

- Address byte savings vs. MATSORT: up to **40%** (avg. **23.9%**)
- Encoding time reduction: **9.2×–24.3×** vs. MATSORT (mean **28.7×**)
- Largest matrix (512×1024, 75% sparsity): **5.13 ms** vs. 95.58 ms for MATSORT
- All outputs verified **lossless** across all 11 test configurations

---

## How It Works

The MAR-MP pipeline consists of four steps:

```
Input Matrix
     │
     ▼
1. Identify Non-Zero Elements
   └─ Record column positions in row-major order
     │
     ▼
2. Delta & NNZ Vector Encoding
   └─ Compute gaps between consecutive non-zero positions
   └─ Record per-row non-zero counts (NNZ vector)
     │
     ▼
3. Offline Merge Planning
   └─ Sort rows by density (descending)
   └─ Greedily pack into merge groups where ΣNNZ ≤ W
   └─ Output fixed (start, length) schedule — no runtime branching
     │
     ▼
4. Indices Encoding at Runtime
   └─ Double-buffered FIFO streams delta sequences
   └─ Position register accumulates deltas per clock cycle
   └─ Count register tracks row boundaries via NNZ vector
   └─ Reconstructed addresses forwarded directly to MAC unit
```

### Delta Encoding Example

For a row with non-zeros at columns `[2, 5, 7]`:
- Stored deltas: `[2, 3, 2]` instead of `[2, 5, 7]`
- At runtime, a single position register accumulates: `0 → 2 → 5 → 7`

### Merge Planning Example

Three rows A, B, C with NNZ = `[3, 2, 3]` and hardware width W = 8:
- 3 + 2 + 3 = 8 ≤ W → all placed in one group: `(start=0, length=3)`
- Hardware follows this schedule directly — no per-row condition evaluation

---

## Installation

```bash
git clone https://github.com/your-org/mar-mp.git
cd mar-mp
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, NumPy, SciPy

---

## Usage

```python
from mar_mp import delta_encode, plan_merge_groups, restore_and_mac

# Compress a sparse matrix
delta_seqs, nnz_vector = delta_encode(sparse_matrix)

# Precompute merge schedule (offline, once per model layer)
merge_groups = plan_merge_groups(nnz_vector, hardware_width=W)

# Reconstruct addresses and perform MAC at inference time
results = restore_and_mac(sparse_matrix, delta_seqs, nnz_vector, merge_groups, input_vector)
```

See `examples/` for full end-to-end demos across different matrix sizes and sparsity levels.

---

## Evaluation

Experiments were run on an AMD Ryzen 5 5600X CPU across **11 matrix configurations** spanning sizes from 64×64 to 512×1024 and sparsity levels from 50% to 90%. Each result is averaged over **200 trials** with randomly generated sparse matrices.

To reproduce results:

```bash
python evaluate.py --configs all --trials 200
```

Metrics reported:
- **Size ratio** — compressed size / original uncompressed size
- **Address metadata bytes** — positional metadata storage only
- **Encoding time (ms)** — software compression time
- **MAR-MP address savings** — % reduction vs. MATSORT

---

## Comparison with Prior Work

| Method         | Conflict-Free | No Residual Zeros | Compact Address | Offline Merge Planning |
|----------------|:---:|:---:|:---:|:---:|
| Column Combine | ✗  | ✓ | ✓ | ✗ |
| HIRAC/SorPack  | ✓  | ✓ | ✗ | ✗ |
| SCNN           | ✗  | ✗ | ✗ | ✗ |
| STPU           | ✓  | ✓ | ✗ | Partial |
| MATSORT        | ✓  | ✓ | ✗ | ✗ |
| dCSR           | —  | — | ✓ | ✗ |
| **MAR-MP**     | ✓  | ✓ | ✓ | ✓ |

---

## Citation

If you use MAR-MP in your research, please cite:

```bibtex
@inproceedings{cho2025marmp,
  title     = {MAR-MP: A Memory-Addressed Register Scheme with Merge Planning
               for Compressed Address Encoding for DNN Applications},
  author    = {Cho, Sungkyun and Ahrar, Alireza and Amirsoleimani, Amirali},
  booktitle = {Proceedings of ...},
  year      = {2025}
}
```

---

## Authors

- **Sungkyun Cho** — Department of Electrical and Computer Engineering, McMaster University, Hamilton, Canada
- **Alireza Ahrar** — Department of Electrical Engineering and Computer Science, York University, Toronto, Canada
- **Amirali Amirsoleimani** — Department of Electrical Engineering and Computer Science, York University, Toronto, Canada

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
