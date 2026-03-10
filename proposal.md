# Project Proposal

## Query-Conditioned Bioacoustic Source Separation

### Applying Query-Based Audio Separation to the Bioacoustic Cocktail Party Problem

---

## 1. Motivation and Problem Statement

Bioacoustic monitoring is a critical tool in ecology, conservation biology, and animal behavior research. A persistent challenge in this field is the **bioacoustic cocktail party problem (CPP)**: isolating the vocalizations of a specific individual animal from recordings that contain overlapping calls from multiple conspecifics. Bermant (2021) addressed this with BioCPPNet, a deep learning architecture applied to macaques, bottlenose dolphins, and Egyptian fruit bats, establishing a strong baseline for single-channel bioacoustic source separation. However, BioCPPNet operates under a fixed-output paradigm — it estimates all sources in a mixture simultaneously, with no mechanism to selectively target a source of interest based on external guidance.

This project proposes to move beyond that paradigm by introducing **query-conditioned separation**: a model that accepts a short reference recording of a target individual and uses it to guide the extraction process, suppressing all other vocalizations. This is a more practical approach for real field deployments, where a reference clip of a focal individual is often available — for example, from a hydrophone attached to a tagged dolphin, or from an earlier clean recording session.

---

## 2. Proposed Approach

### 2.1 Core Idea

The central idea is straightforward: rather than asking the model to separate everything at once, we give it a hint. A short enrollment clip from the individual of interest is encoded into an embedding vector that describes the acoustic character of that individual's vocalizations. This embedding is then used to condition the separation network, steering it toward extracting the matching source and suppressing others.

This approach draws on recent advances in **query-based music source separation**, where the same concept has been successfully applied to isolate arbitrary instrument types from musical mixtures (Watcharasupat & Lerch, 2024). The core machinery — a bandsplit encoder-decoder conditioned via a Feature-wise Linear Modulation (FiLM) layer — is well-understood, openly available, and directly transferable to the bioacoustic domain.

### 2.2 Implementation Plan

We will build on the publicly available **Banquet codebase** ([github.com/kwatcharasupat/query-bandit](https://github.com/kwatcharasupat/query-bandit)), which implements the query-based music separation system from Watcharasupat & Lerch (2024). This gives us a solid, tested foundation rather than building from scratch.

The adaptation to bioacoustics involves three focused steps:

1. **Data pipeline.** Replace the music-domain data pipeline with bioacoustic recordings: resampling to the required format, synthesizing two-speaker mixtures from clean individual recordings, and extracting held-out enrollment clips per individual.
2. **Query encoding.** The Banquet system uses a pretrained audio embedding model (PaSST) to encode query clips into a compact vector representation. We will evaluate whether this general-purpose embedding transfers effectively to bioacoustic signals, and fine-tune or replace it if needed.
3. **Conditioning and training.** The FiLM conditioning module and bandsplit encoder-decoder remain structurally unchanged. Training adapts the model to bioacoustic mixture statistics, using individual identity as the supervision signal.

### 2.3 Potential Extension: Region-Based Queries

A natural limitation of point-based query conditioning is that a single embedding vector may not capture the full variability of an individual's vocalizations — for example, an animal that produces several distinct call types. Recent work by the same authors (Watcharasupat & Lerch, 2025) proposes representing the query as a **region in embedding space** (specifically, a hyperellipsoid) rather than a single point. This allows the model to target a broader acoustic neighborhood, accommodating intra-individual variability.

If the core system performs well and time permits, we will explore this extension as a second phase of the project. It requires augmenting the query representation and adjusting the input to the FiLM mapping network, with no changes to the rest of the architecture. This is a meaningful research contribution if it works, but it is not a prerequisite for the project to deliver value.

---

## 3. Baselines

Performance will be evaluated against three reference points:

- **BioCPPNet** (Bermant, 2021): the established permutation-invariant baseline, representing the current state of the art for blind bioacoustic separation.
- **Conv-TasNet with dot-product attention** (Luo & Mesgarani, 2019): a lighter-weight query-conditioned baseline, representing a simpler form of guided separation without the bandsplit structure.
- **Banquet point-query model adapted to bioacoustics** (this work, Stage 1): the direct output of adapting the music separation system, used as an internal checkpoint before any further extensions.

The primary metrics are SI-SDR (scale-invariant signal-to-distortion ratio) and SI-SDRi (SI-SDR improvement over the unprocessed mixture), consistent with the BioCPPNet evaluation protocol. Individual identity classification accuracy on the separated outputs will serve as a secondary, task-specific metric.

---

## 4. Data

We will use the same datasets as Bermant (2021) to ensure direct comparability:

- **Bottlenose dolphin signature whistles** — the primary focus species, given the high individual distinctiveness of signature whistles and their relevance to tagging-based research scenarios.
- **Macaque vocalizations** and **Egyptian fruit bat vocalizations** — as secondary species for generalization analysis, time permitting.

Artificial two-speaker mixtures will be synthesized by summing clean single-individual recordings at randomized SNR offsets (0–5 dB), following the BioCPPNet protocol. Enrollment clips are drawn from held-out segments of the same individual, not present in the mixture.

---

## 5. Scope and Deliverables

This project is scoped as a class assignment with a defined timeline. The primary deliverable is a **working query-conditioned separation model for bottlenose dolphin vocalizations**, validated against the BioCPPNet baseline. The hyperellipsoidal extension is an exploratory secondary goal, pursued only if the core system is stable and performing well.

|Deliverable|Status|
|---|---|
|Banquet codebase adapted to bioacoustic data|Core deliverable|
|Quantitative comparison vs. BioCPPNet and Conv-TasNet|Core deliverable|
|Qualitative failure case analysis|Core deliverable|
|Hyperellipsoidal query extension|Stretch goal|

The expected outcome is a demonstration that query-conditioned separation — a paradigm already proven in music — can be meaningfully transferred to bioacoustic signals, and a concrete assessment of where the approach succeeds and where it faces domain-specific challenges.

---

## References

Bermant, P. C. (2021). BioCPPNet: automatic bioacoustic source separation with deep neural networks. _Scientific Reports_, 11, 23502.

Luo, Y., & Mesgarani, N. (2019). Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_, 27(8), 1256–1266.

Watcharasupat, K. N., & Lerch, A. (2024). A stem-agnostic single-decoder system for music source separation beyond four stems. _Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR)_. arXiv:2406.18747.

Watcharasupat, K. N., & Lerch, A. (2025). Separate This, and All of These Things Around It: Music Source Separation via Hyperellipsoidal Queries. arXiv:2501.16171.