# RoPE-Aware Attribution with Cosine Semantic Scoring

**Effective Context Utilization in Large Language Models**  
University of Illinois Urbana-Champaign — CS 498: Machine Learning Systems (May 2026)

---

## Overview

This project investigates the *lost-in-the-middle* phenomenon in transformer-based LLMs: retrieval accuracy degrades for information near the center of long inputs. We implement and evaluate three **RoPE-safe attribution methods** on a needle-in-a-haystack (NIAH) benchmark, introduce new positional bias metrics, and propose a cosine-semantic importance variant that captures output degradation below the exact-match threshold.

**Key insight:** Token deletion and 1-D padding masks both corrupt RoPE positional encodings. All three methods in this work preserve position indices throughout attribution.

---

## Methods

| ID | Method | RoPE-safe | Cost |
|---|---|---|---|
| A | Substitution Ablation | ✓ | O(N_chunks) |
| B | InputXGrad | ✓ | O(1) |
| C | 4-D Causal Block Masking | ✓ | O(N_chunks) |
| — | Token deletion (baseline, invalid) | ✗ | O(N_chunks) |
| — | 1-D padding mask (baseline, invalid) | ✗ | O(N_chunks) |

**Method A (Substitution Ablation):** Replaces all token IDs in a chunk with the period token, preserving sequence length and all RoPE indices.

**Method B (InputXGrad):** Single forward-backward pass. Token importance = `||∇_e L ⊙ e||₂`. Requires `bfloat16` to avoid silent NaN propagation through RMSNorm layers (see Engineering Notes).

**Method C (4-D Causal Block Masking):** Adds a 4-D additive mask (`-inf`) to pre-softmax logits *after* RoPE rotations, blocking attention to a target chunk column while preserving causal structure and all positional encodings.

---

## Metrics

| Metric | What it measures |
|---|---|
| **NCR** (Needle Chunk Rank) | Rank of the needle chunk in the importance vector (lower = better; 1 is optimal) |
| **NCR₋q** | NCR with question chunk zeroed out |
| **NRD** | Needle rank within document region only (doc-only mode) |
| **NRD_cos** | NRD under cosine semantic scoring |
| **EUCR@λ** | Fraction of chunks with normalized importance > λ (context breadth) |
| **PWUP** | Mean normalized importance per context third (Beginning / Middle / End) |
| **GUD** | Spearman ρ between chunk position and importance (+ = recency bias, − = primacy bias) |

### Cosine Semantic Importance

For perturbation methods (A and C), importance is also scored as:

```
imp_cos(c_i) = 1 - cos_sim( φ(o_base), φ(o_i) )
```

where `φ` is `all-MiniLM-L6-v2` (384-dim, CPU). Captures partial output degradation below the exact-match threshold.

### Doc-Only Ablation Mode

Sets the question chunk importance to 0 before ranking, removing systematic question-chunk bias from needle rank computation. Produces NRD and NRD_cos variants.

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Model | `Qwen2.5-1.5B-Instruct` |
| Context lengths | 512, 1024, 2048 tokens |
| Needle depths (δ) | 0.1, 0.25, 0.5, 0.75, 0.9 |
| Examples per config | 2 |
| Total prompts | 30 (3 × 5 × 2) |
| Chunk size B | 128 tokens |
| Max new tokens | 64 (greedy decoding) |
| EUCR thresholds | 0.01, 0.05, 0.10 |
| Hardware | Single NVIDIA H200 GPU |
| Precision | bfloat16 |

**Needle types** (runtime-generated, not memorizable):
- UUID: `"The secret activation code is: X7QK3P1A."`
- Key-value: `"system_port = 4821"`

**Haystack:** Repeating scientific-text paragraph, trimmed to exact token budget. Needle inserted at word index `⌊δ · |W|⌋`.

---

## Results Summary

### Needle Chunk Rank (NCR) vs. Context Length

| Method | 512 tok | 1024 tok | 2048 tok |
|---|---|---|---|
| Substitution (A) | 3.4 | 5.5 | 8.1 |
| InputXGrad (B) | 3.7 | 6.1 | 11.1 |
| Mask4D (C) | 3.0 | 5.9 | 9.6 |

NCR approaches near-random by 2048 tokens (~16 total chunks) for all methods.

### Positional Bias (PWUP_E ≫ PWUP_M for all methods → strong recency bias)

GUD is positive for all three methods:
- Method C (Mask4D): ρ = 0.28 (strongest recency tendency)
- Methods A and B: ρ ≈ 0.11–0.12

Method C is a notable exception to the U-shaped profile: `PWUP_M > PWUP_B` at 512 and 1024 tokens, suggesting 4-D masking captures genuine middle-region causal influence that substitution misses.

### Key Findings

- Evidence localization degrades consistently with context length across all methods.
- Cosine scoring improves needle rank over discrete scoring for perturbation methods (e.g., NRD 2.9 → 2.6 for Method A at 512 tokens; 8.3 → 7.7 for Method C at 2048 tokens).
- Doc-only correction reduces question-chunk bias most significantly for InputXGrad (NCR 3.7 → 2.7 at 512 tokens).
- InputXGrad degrades fastest at long contexts due to gradient diffusion over softmax distributions.

---

## Engineering Notes

**bfloat16 required for InputXGrad.**  
`float16` causes silent NaN hidden states in Qwen2.5's RMSNorm layers. Outputs appear normal but gradients are all zero — detectable only via the GUD column. `bfloat16` eliminates this.

**Chat template is mandatory.**  
Omitting `<|im_start|>` / `<|im_end|>` wrappers substantially reduces baseline accuracy. The model was fine-tuned exclusively on templated data.

**Specify both EOS token IDs.**  
Must set `eos_token_id` to both `151643` (`<|endoftext|>`) and `151645` (`<|im_end|>`). Without this, the model runs to `max_new_tokens` and trailing garbage tokens break substring-match evaluation.

**Prefill-then-decode loop required for Method C.**  
`model.generate()` rebuilds the attention mask at every decode step, overwriting the 4-D block mask. A manual prefill → token-level decode loop is required to preserve the mask throughout generation.

---

## Dependencies

```
torch>=2.0          # bfloat16, eager attention
transformers>=4.0   # Qwen2.5 support
sentence-transformers  # all-MiniLM-L6-v2
numpy
scipy               # Spearman correlation for GUD
```

Model: `Qwen/Qwen2.5-1.5B-Instruct` (HuggingFace Hub)  
Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (loaded on CPU to preserve GPU VRAM)

---

## Limitations

- Single model (Qwen2.5-1.5B). Larger models (7B, 72B) may exhibit different bias magnitudes.
- Synthetic repeating-paragraph haystack. Real document structure may redistribute importance differently.
- Maximum context of 2048 tokens. Commercially relevant scenarios operate at 32k–128k+.
- 2 seeds per (length, depth) cell — insufficient for significance testing.
- Attention aggregation method fully implemented but excluded from main runs.
- InputXGrad not compared against full Integrated Gradients.

---

## Citation

```bibtex
@article{kadam2026rope,
  title   = {Effective Context Utilization in Large Language Models:
             RoPE-Aware Attribution with Cosine Semantic Scoring},
  author  = {Kadam, Advay and Sapre, Aryan and Raina, Naman},
  year    = {2026},
  note    = {CS 498: Machine Learning Systems, UIUC}
}
```
