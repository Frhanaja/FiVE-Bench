[![Releases](https://img.shields.io/badge/Releases-v1.0-blue?logo=github&style=for-the-badge)](https://github.com/Frhanaja/FiVE-Bench/releases)

# FiVE-Bench: Fine-grained Video Editing Benchmark for ICCV 2025

[![Topics](https://img.shields.io/badge/topics-rectified--flow%20%7C%20text--to--video--dataset%20%7C%20video--editing%20%7C%20video--gen-green?style=for-the-badge)](https://github.com/Frhanaja/FiVE-Bench)
![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=for-the-badge)
![ICCV 2025](https://img.shields.io/badge/ICCV-2025-red?style=for-the-badge)

🧭 A benchmark and toolkit for testing fine-grained video editing with diffusion and rectified flow models. This repo supports dataset access, evaluation scripts, baseline models, and a reproducible protocol. Get the release bundle and run the reference setup: https://github.com/Frhanaja/FiVE-Bench/releases

![Banner: Video edit pipeline](https://images.unsplash.com/photo-1519389950473-47ba0277781c?ixlib=rb-4.0.3&w=1400&q=80)

Table of contents
- [Overview](#overview)
- [Why FiVE-Bench](#why-five-bench)
- [Key features](#key-features)
- [Dataset](#dataset)
  - [Design goals](#design-goals)
  - [Splits and scale](#splits-and-scale)
  - [Annotation format](#annotation-format)
  - [Example data](#example-data)
- [Tasks and challenge](#tasks-and-challenge)
  - [Edit types](#edit-types)
  - [Evaluation protocol](#evaluation-protocol)
- [Metrics](#metrics)
- [Baselines and models](#baselines-and-models)
- [Releases and downloads](#releases-and-downloads)
- [Quickstart](#quickstart)
  - [Environment](#environment)
  - [Install from release](#install-from-release)
  - [Run a baseline](#run-a-baseline)
  - [Evaluate outputs](#evaluate-outputs)
- [Repository layout](#repository-layout)
- [Detailed pipeline](#detailed-pipeline)
- [Annotation schema](#annotation-schema)
- [Reproducibility notes](#reproducibility-notes)
- [Leaderboard example](#leaderboard-example)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact & maintainers](#contact--maintainers)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)

## Overview

FiVE-Bench provides a controlled and diverse set of video editing cases. The data focuses on fine-grained edits that require spatial, temporal, and semantic coherence. We test emerging diffusion models and rectified flow models under unified protocols. The repo includes:

- A curated video dataset with target edits and source videos.
- JSON annotations for edit instructions, masks, and ground truth.
- Baseline implementations: text-guided diffusion, mask-conditioned rectified flow, and hybrid models.
- Evaluation tools: FID, CLIP-based perceptual metrics, LPIPS, temporal consistency, and user study scripts.
- Reproducible configs and Docker images.

FiVE-Bench aims to measure model behavior on subtle edits such as object color change, motion re-timing, style transfers limited to regions, and structured content insertion.

## Why FiVE-Bench

Researchers need benchmarks that stress models on real editing demands. Existing text-to-video sets focus on generation from scratch. FiVE-Bench focuses on editing an existing clip while preserving structure. The benchmark targets:

- Fine-grained control: edits limited to regions or frames.
- Temporal coherence: maintain motion and identity across frames.
- Cross-modal semantics: use text prompts and masks together.
- Evaluation that matches human judgement and perceptual metrics.

## Key features

- Multi-modal annotations: text prompts, masks, keyframes, and auxiliary metadata.
- Multiple difficulty levels: from simple color swap to complex object insertion.
- Baselines for both diffusion and rectified flow families.
- Standardized evaluation and leaderboard tools.
- Release bundle with scripts, preprocessed data samples, and Docker images.

## Dataset

### Design goals

FiVE-Bench follows three main goals:

1. Control. Provide mask and keyframe guidance so models must follow constraints.
2. Realism. Use real-world video sources to keep motion and lighting challenging.
3. Diversity. Include varied object classes, motions, and scenes.

### Splits and scale

- Train: 5,200 short clips (3-6 seconds) with paired edit targets.
- Validation: 800 clips.
- Test: 1,000 clips (held out for leaderboard).
- Each clip includes 16-32 frames at 256–512 px short edge.

The dataset includes balanced categories: object edits, scene edits, attribute edits, and compositional edits.

### Annotation format

Each clip has a JSON entry with the following fields:

- id: unique id
- src_video: path to source video frames
- target_video: path to target frames (ground truth after edit)
- prompt: text instruction for edit
- mask: path to binary mask per frame (or single mask with transform)
- keyframes: list of frame indices with stronger constraints
- difficulty: enum {easy, medium, hard}
- meta: dict with fps, resolution, object classes, and bounding boxes

See the [Annotation schema](#annotation-schema) for full example.

### Example data

Preview frames and masks appear in the docs folder and the release bundle. Example images:

![Sample edit 1](https://images.unsplash.com/photo-1532619675605-8f8ad9f2f2b5?w=800&q=80)
![Sample edit 2](https://images.unsplash.com/photo-1504203700689-24f7f6b8a1d6?w=800&q=80)

These images show representative scenes. The release contains full clips with masks and prompts.

## Tasks and challenge

FiVE-Bench defines several tasks that reflect common editing demands.

### Edit types

- Attribute edit: change color, texture, or material of an object.
- Localized style transfer: apply a painterly or photoreal style only on the subject.
- Object manipulation: remove, replace, or re-position an object.
- Motion edit: alter motion speed or direction for a subject.
- Compositional edit: combine multiple edits in sequence.

Each task has explicit constraints. Some tasks require high temporal fidelity. Others test cross-frame semantic consistency.

### Evaluation protocol

- Use the provided preprocess script to standardize frames.
- Run the model on each test clip with the supplied prompt and mask.
- Submit an output folder that matches the test structure: id/{frames}.
- Run the eval script to compute metrics.
- For leaderboard entries, also submit raw model outputs and a runlog file with seed and config.

Driven by reproducibility, each run must set a fixed random seed and record system hardware.

## Metrics

We include objective and human-centered metrics.

Objective metrics
- FID (per-frame and aggregated): measure distribution gap between generated and ground truth frames. Compute per-clip and global FID.
- CLIP-Similarity: compute CLIP image-text score between generated clip and prompt. Aggregate across frames with median and percentile.
- LPIPS: perceptual distance across frames.
- Temporal Consistency (TC): measure pixel or feature drift across adjacent generated frames vs ground truth.
- Mask IoU: measure mask adherence when the edit is mask-guided.
- Object Identity Score (OID): feature-based cosine on detected object embeddings across frames to measure identity preservation.

Human study metrics
- Alignment: How well does the clip match the prompt (1-5 scale).
- Naturalness: Does the clip look real and coherent (1-5 scale).
- Temporal smoothness: perceived jitter or flicker (1-5 scale).
- Preference: head-to-head A/B for top submissions.

Evaluation details
- For CLIP-based metrics, use CLIP ViT-B/32 or ViT-L/14 depending on resolution. Report both.
- Compute FID on pooled frames and per-category frames for targeted analysis.
- For temporal metrics, use features from a video-based network (e.g., I3D) for stability measures.

## Baselines and models

We include reference implementations and configs. Key families:

- Text-to-video diffusion (frame-wise + temporal conditioning)
  - StableDiffusion-based frame editor with temporal smoothing.
  - T2V diffusion model with cross-frame attention.

- Rectified flow editors
  - Mask-conditioned rectified flow that modifies pixels with constraints.
  - Frame-inpainting flow that uses optical flow guidance.

- Hybrid models
  - Diffusion front-end with rectified flow temporal refiner.

Each baseline comes with:
- Pretrained checkpoints for short runs.
- Training scripts with common hyperparameters.
- Inference scripts tuned for GPU and CPU.

Performance tuning
- Use 8-bit Adam or AdamW with grad accumulation for larger batches.
- Use mixed precision (fp16) with gradient scaling on GPUs.

Model cards
- Each baseline includes a model card with training data, compute, and known failure modes.

## Releases and downloads

Download the official release bundle from the Releases page. The release includes dataset samples, baseline checkpoints, Docker images, and the reference run script.

Download and execute the provided setup file from the releases: https://github.com/Frhanaja/FiVE-Bench/releases

The release bundle contains:
- FiVE-Bench-release_v1.0.zip
- setup.sh (install dependencies and download required models)
- docker/FiVE-Bench.Dockerfile
- samples/ (examples and small dataset sample)
- baselines/ (inference and training scripts)
- eval/ (metric scripts and runlog tools)

To install from the release after download:
- Unpack the zip.
- Run setup.sh to prepare the environment and fetch model checkpoints.
- The setup script can create a conda env, install pip packages, and pull Docker images.

If the release link fails, check the Releases section on GitHub.

[![Download Releases](https://img.shields.io/badge/Download-Releases-blue?logo=github&style=for-the-badge)](https://github.com/Frhanaja/FiVE-Bench/releases)

## Quickstart

This quickstart shows the minimal end-to-end flow. It uses the release bundle and a baseline inference script.

### Environment

- Linux or macOS
- GPU recommended (NVIDIA with CUDA 11+)
- Python 3.10 or 3.11
- Docker optional for isolation

Recommended package set
- torch >= 2.0
- torchvision
- diffusers >= 0.15
- accelerate
- opencv-python
- lpips
- pytorch-ignite (for logging)
- ftfy, regex, transformers, tokenizers
- scikit-image, scipy

### Install from release

1. Download the release archive from the Releases page:
   - Visit https://github.com/Frhanaja/FiVE-Bench/releases and download FiVE-Bench-release_v1.0.zip

2. Unpack and run setup:
   - unzip FiVE-Bench-release_v1.0.zip
   - cd FiVE-Bench-release_v1.0
   - bash setup.sh

setup.sh handles:
- Creating a conda env or virtualenv
- Installing Python packages
- Downloading pretrained checkpoints
- Preparing a small sample dataset in data/sample

If the release link does not work, open the repository Releases section on GitHub.

### Run a baseline

Run the mask-guided diffusion baseline on a sample clip:

- Command example:
  - python baselines/diffusion_infer.py \
    --config baselines/configs/diffusion_mask.yaml \
    --input data/sample/src_clip \
    --mask data/sample/mask \
    --prompt "Make the red jacket green while keeping motion" \
    --output out/diffusion_out \
    --seed 42

This script reads frames, applies the model, and writes frames to out/diffusion_out. The config file sets model size, guidance scale, and temporal smoothing weight.

For rectified flow baseline:

- python baselines/rectified_flow_infer.py \
  --config baselines/configs/rectified_flow.yaml \
  --input data/sample/src_clip \
  --mask data/sample/mask \
  --prompt "Replace bicycle with skateboard" \
  --output out/flow_out \
  --seed 42

### Evaluate outputs

Use the evaluation tool to compute the benchmark metrics:

- python eval/eval_all.py \
  --pred out/diffusion_out \
  --gt data/sample/target_clip \
  --mask data/sample/mask \
  --out out/eval_report.json

The script computes FID, CLIP, LPIPS, TC, and returns a JSON report. For cross-submission leaderboard entries, run with `--detailed` to log per-frame metrics.

## Repository layout

A high-level look at files in the release (and main repo):

- baselines/
  - diffusion_infer.py
  - rectified_flow_infer.py
  - train_diffusion.py
  - configs/
- data/
  - sample/
  - preprocess.py
- eval/
  - eval_all.py
  - metrics/
    - fid.py
    - clip_sim.py
    - lpips.py
    - temporal_consistency.py
- docker/
  - FiVE-Bench.Dockerfile
- docs/
  - examples/
  - images/
- models/
  - checkpoints/ (downloaded via setup)
- scripts/
  - setup.sh
  - export_videos.py
- LICENSE
- CITATION.bib
- README.md

## Detailed pipeline

The benchmark pipeline follows these steps:

1. Data prep
   - Use preprocess.py to extract frames, normalize sizes, and generate masks for complex cases.
   - Store frames as PNGs in a directory per clip.

2. Inference
   - Provide prompt, mask, and optional keyframe constraints.
   - The model processes frames in sliding windows to manage memory.
   - Use temporal attention or flow-based conditioning to preserve motion.

3. Post-processing
   - Blend generated pixels with original frames using the mask.
   - Apply color correction to match global histograms of the source clip.
   - Export frames to video with ffmpeg.

4. Evaluation
   - Align frames with ground truth and compute per-frame and per-clip metrics.
   - Run user study tools for subjective metrics.

Implementation notes
- We implement a frame bank to cache features for adjacent frames.
- The rectified flow model uses a forward-backward solver to integrate constraints.
- We provide a temporal refinement pass to reduce flicker.

## Annotation schema

Sample JSON for one clip:

{
  "id": "FB_000123",
  "src_video": "data/src/FB_000123/frames",
  "target_video": "data/target/FB_000123/frames",
  "prompt": "Change the blue car to matte red, keep reflections",
  "mask": "data/masks/FB_000123/mask_%04d.png",
  "keyframes": [0, 8, 15],
  "difficulty": "medium",
  "meta": {
    "fps": 24,
    "resolution": "480x270",
    "objects": ["car"],
    "bbox": [[12,34,160,90]]
  }
}

Mask convention
- Masks are binary PNGs. White pixels indicate edit region. Masks can vary per frame.
- If a single mask is provided, the model should warp the mask by tracking object motion.

## Reproducibility notes

- Record random seed, PyTorch version, CUDA version, and GPU model.
- Use deterministic dataloaders when possible.
- For deterministic sampling in diffusion models, set deterministic sampler flags and save RNG state.

Hardware logs
- The reference runs include a runlog with:
  - seed, git commit, GPU model, CUDA, cudnn, python version, conda env hash.

Training notes
- For large models use gradient checkpointing and activation recomputation.
- We recommend training with temporally shuffled batches and explicit windowing.

## Leaderboard example

We host a baseline leaderboard for submissions. The format for a submission:

- Submission bundle:
  - outputs.zip (one folder per clip with frames)
  - eval_report.json
  - runlog.json
  - model_card.md

Example leaderboard columns:
- Rank
- Method
- FID (↓)
- CLIP (↑)
- LPIPS (↓)
- TC (↑)
- Human score (↑)

Sample table (values synthetic):

| Rank | Method | FID | CLIP | LPIPS | TC | Human |
|------|--------|-----:|-----:|-----:|----:|-----:|
| 1 | Diffusion+FlowRefine | 18.4 | 0.72 | 0.081 | 0.89 | 4.2 |
| 2 | RectifiedFlow-Mask | 21.7 | 0.69 | 0.095 | 0.86 | 4.0 |
| 3 | Frame-wise Diffusion | 28.9 | 0.64 | 0.12 | 0.78 | 3.6 |

This table illustrates typical performance tradeoffs.

## Contributing

We welcome contributions that improve the benchmark or add baselines.

How to contribute
- Fork the repo.
- Create a feature branch.
- Add tests or examples for new code.
- Open a pull request with a clear description and rationale.

Guidelines
- Keep API stable for eval scripts.
- Add config files for new baselines under baselines/configs.
- Update CITATION.bib when adding references.

Reporting issues
- Open issues for bugs, metric inconsistencies, or data problems.
- Provide minimal repro steps and relevant logs.

Code style
- Follow PEP8 where possible.
- Keep functions short and focused.
- Write docstrings for major functions.

## Citation

If you use FiVE-Bench, cite the ICCV 2025 paper:

Please cite:

Frhanaja, A., et al. "FiVE-Bench: A Fine-grained Video Editing Benchmark for Evaluating Emerging Diffusion and Rectified Flow Models." ICCV 2025.

CITATION.bib (example)
@inproceedings{frhanaja2025five,
  title={FiVE-Bench: A Fine-grained Video Editing Benchmark for Evaluating Emerging Diffusion and Rectified Flow Models},
  author={Frhanaja, A. and Others},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2025}
}

## License

FiVE-Bench uses the Apache 2.0 License. See the LICENSE file for full text. The dataset samples in the release follow usage terms included in the data_readme.txt in the release bundle.

## Contact & maintainers

Maintainers
- A. Frhanaja — lead author (email in repo)
- Bench team — bench-maintainers@org.example

For faster response
- Open an issue on GitHub for bugs and feature requests.
- Use PRs for code changes.

Community
- Join discussions on the Issues and Discussions tab.
- For major changes, propose RFC via an issue.

## FAQ

Q: Where do I get the dataset?
A: Download the release bundle and run setup.sh. The release includes a small sample and scripts to fetch the full data if you have access rights.

Q: How do I reproduce numbers in the paper?
A: Use the provided baselines with their config files, fix the seed, and set the same runtime flags listed in runlog files in the release.

Q: Can I submit to the leaderboard?
A: Yes. Follow the submission format in docs/leaderboard.md and include runlog.json and eval_report.json.

Q: The release link does not work. What now?
A: Check the Releases section of the GitHub repo. If the release is missing, open an issue.

Q: Do you provide Docker containers?
A: The release includes a Dockerfile and prebuilt images in the release bundle for supported platforms.

## Acknowledgements

We thank the open-source communities that support model libraries and metrics. We use several public datasets for training baselines and provide attribution in the model cards. The project uses open licenses and encourages reuse and extension.

END OF FILE