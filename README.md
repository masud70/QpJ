# Quality per Joule

Replication package for the paper **"Does One Model Rule Them All? Energy vs. Quality in Image Generation Models"** (submitted to GREENS 2027).

A controlled comparison of the energy and output quality of five image generation models from three families (AE, VAE, DCGAN, WGAN-GP and DDPM) on CelebA at 32×32.

Every model is trained with the objective of its original publication. AE, VAE, DCGAN and WGAN-GP share the DCGAN architecture; DDPM keeps its original U-Net. Energy is measured per segment for CPU, RAM and GPU, using one of three interchangeable backends.

## Reproducing the paper

All numbers, tables and figures in the paper are produced by [`analysis.ipynb`](analysis.ipynb) from the four result folders below. Every folder holds 5 models × 3 seeds (42, 1337, 2026).

| Training budget | Native | CodeGreen |
|---|---|---|
| Fixed budget (30 epochs) | `results/fixed_budget_ddim100__native` | `results/fixed_budget_ddim100_final__codegreen` |
| Quality target (FID ≤ 50, cap 100 epochs) | `results/target_quality_fid50_ddim100_final__native` | `results/target_quality_fid50_ddim100_final__codegreen` |

In all four folders, DDPM uses DDIM-100 for the periodic FID checks, the final sampling and final FID, and the inference benchmark (see [DDIM](#stopping-criteria)).

**Hardware and software used.** Intel Core i7-9700K, 32 GB RAM, NVIDIA GeForce RTX 2080 SUPER (250 W power limit), Ubuntu 22.04, PyTorch 2.7.1 with CUDA 11.8. Absolute energy values depend on the hardware and the measurement tool; compare them only within one setup.

**Running the notebook.** Open it from the repository root, next to `results/`. Besides `requirements.txt`, it needs `pandas`, `scipy` and `jupyter`:

```bash
source .venv/bin/activate
pip install pandas scipy jupyter
jupyter notebook analysis.ipynb
```

## Design

**Shared backbone.** The AE/VAE decoder and both GAN generators are the same Radford et al. (2016) generator at 32×32 (channel widths 256 → 128 → 64). The AE/VAE encoder, the DCGAN discriminator and the WGAN-GP critic are the same mirrored encoder. Energy differences between these four models therefore reflect the training paradigm rather than the network. The original AE and VAE networks are small fully connected models designed for MNIST and are not competitive on 32×32 colour faces, so they are replaced by the shared convolutional backbone (as in Ghosh et al. 2020). DDPM cannot share a GAN backbone and keeps its original U-Net; its larger capacity is reported as parameters and FLOPs.

| Model | Architecture | Arch. source | Objective | Optimizer | Reference |
|---|---|---|---|---|---|
| AE | Shared encoder/decoder, z = 128. Samples are drawn from a full-covariance Gaussian fitted to the training latents. | Shared (DCGAN); original was fully connected | Summed squared error | Adam 2e-4, β (0.5, 0.999) (DCGAN settings) | Hinton & Salakhutdinov 2006; Ghosh et al. 2020 |
| VAE | Shared encoder (μ, log σ²) and decoder | Shared (DCGAN); original was fully connected | Gaussian NLL (σ² = 0.5, matching the AE's squared error) + KL, β = 1 | Adam 2e-4, β (0.5, 0.999) (DCGAN settings) | Kingma & Welling 2014 |
| DCGAN | Generator and discriminator (BatchNorm) | Original, scaled to 32×32 | Non-saturating BCE | Adam 2e-4, β (0.5, 0.999) | Radford et al. 2016 |
| WGAN-GP | Shared generator; critic uses LayerNorm | Shared (DCGAN); the paper's DCGAN-style variant, LayerNorm as recommended | Wasserstein + GP, λ = 10, n_critic = 5 | Adam 1e-4, β (0, 0.9) | Gulrajani et al. 2017 |
| DDPM | U-Net with 35.7M parameters (ch 128, mult 1,2,2,2, attention at 16×16, dropout 0.1) | Original (CIFAR-10 configuration) | ε-MSE, T = 1000, linear β schedule | Adam 2e-4, 5k-step warmup, gradient clip 1, EMA 0.9999 with warmup min(0.9999, (1+n)/(10+n)) | Ho et al. 2020 |

**Held constant across all models:** dataset and preprocessing (178-pixel center crop, resized to 32×32), training batch size (32 in the paper; the config default is 64), FP32, seeds, hardware and the FID protocol. Image generation (FID checks, final sampling and inference) uses batches of 250.

## Setup

On the Linux GPU machine, run:

```bash
bash setup.sh
```

The script is safe to re-run. Steps that are already done are skipped. It does the following:

1. Checks the Python version, the GPU and free disk space (about 15 GB is needed).
2. Creates `.venv` and installs PyTorch for CUDA 11.8 plus `requirements.txt`. For a different CUDA build, set `TORCH_INDEX=https://download.pytorch.org/whl/<cuXXX>`.
3. Makes the RAPL energy counters readable with `sudo chmod`. This lasts until the next reboot, so re-run `setup.sh` after rebooting. Nothing permanent is changed on the system. In particular, `codegreen init-sensors` is not used, because it installs a udev rule and a user group.
4. Downloads CelebA and builds the caches (`python run_all.py prepare`).
5. Runs `check` for every backend and saves the output to `results/setup_checks/<backend>.txt`.

**Dataset.** The official aligned and cropped CelebA images are downloaded from the Hugging Face mirror [`flwrlabs/celeba`](https://huggingface.co/datasets/flwrlabs/celeba). The files are PNGs in Parquet format, using the official splits: 162,770 train and 19,962 test. The download is pinned to revision `data.hf_revision`, which makes it byte-identical on every machine. Only the train and test shards are downloaded (10.6 GB) into `data/celeba/parquet/`. Any later command that needs the data downloads it automatically if it's missing.

## Usage

```bash
source .venv/bin/activate
python run_all.py check   --backend native                        # confirm which components are measured
python run_all.py prepare                                         # download data, cache 32×32 + FID reference statistics
python run_all.py train   --backend native --stopping fixed_budget # all models × all seeds
python run_all.py analyze --backend native --stopping fixed_budget # tables and plots
```

`setup.sh` already runs `check` and `prepare`. For the paper's exact settings, see [Reproducing the paper](#reproducing-the-paper).

| Flag | Values | Default |
|---|---|---|
| `--backend` | `native`, `codegreen` | `energy.backend` (`native`) |
| `--stopping` | `fixed_budget`, `target_quality` | `stopping.mode` (`fixed_budget`) |
| `--target-fid` | number; `target_quality` only | `stopping.target_quality.target_fid` (25; the paper uses 50) |
| `--ddim` | optional step count; bare flag = 100 | `ddpm.ddim_steps` (off; the paper uses 100) |
| `--models` | any of `ae vae dcgan wgan_gp ddpm` | all five |
| `--seeds` | integers, space separated | `42 2026 1337` |
| `--config` | YAML file merged over `config/default.yaml` | none |
| `--set` | `key.path=value`, repeatable | none |
| `--overwrite` | re-run finished runs | off |

Each (model, seed) run executes in its own subprocess. Runs that already finished are skipped, so an interrupted suite can simply be restarted. Runs that failed or were interrupted are re-run from the start. Before any training starts, the suite checks every finished run's `config.yaml` against the current config. If a setting that affects results has changed, the suite stops and asks for `--overwrite` or a different `experiment.results_dir`.

```bash
python run_all.py train --backend codegreen --stopping target_quality --models dcgan ddpm --seeds 42
python run_all.py train --config config/smoke.yaml --backend native     # quick CPU test, synthetic data
```

## Energy measurement

Only the selected backend runs. Every backend reports the same fields for each segment: `cpu_j`, `ram_j`, `gpu_j`, `total_j = cpu + ram + gpu`, `duration_s`, and `measured` flags that show which values were measured rather than estimated. The paper reports gross energy (`total_j`).

| | `native` (standard library only) | `codegreen` |
|---|---|---|
| CPU | RAPL `package-*` (sysfs) | RAPL package |
| RAM | RAPL `dram`; estimated from `energy.ram_w_per_gb` if absent | RAPL `dram`; estimated if absent |
| GPU | NVML total-energy counter (ctypes) | NVML |

**Segments.** `idle` (baseline, used to compute `net_total_j`), `train_<k>`, `gaussian_fit` (AE only, counted as training), `eval_<k>` (target quality only), `final_<sampler>` and `inference_<sampler>`.

**Inference benchmark.** Each model generates images in batches of `evaluation.batch_size` (250) until it has produced at least `evaluation.inference_images` (1,000) images and sampled for at least `evaluation.inference_min_s` (30 s). Energy is reported per 1,000 images.

## Stopping criteria

- **`fixed_budget`:** stop at `budget`, given in `epochs | steps | images | wall_clock_s | energy_j`. Training is measured in segments of size `segment`. Energy budgets are checked at segment ends, so a run can overshoot by at most one segment.
- **`target_quality`:** every `eval_every`, compute FID on `fid_samples` images. Stop when FID ≤ `target_fid` (status `target_reached`), or stop at `max_budget` (status `budget_exhausted`).

**DDIM (DDPM only).** By default, DDPM samples ancestrally (1,000 steps). With `--ddim N`, DDPM's periodic FID checks use DDIM-N (η = 0), and the inference benchmark also reports DDIM-N; training is unchanged. `train` itself draws the final samples ancestrally. In the result folders used by the paper, DDPM's final sampling and final FID were regenerated with DDIM-100 from the saved final checkpoints and metered in the same way, so DDPM uses DDIM-100 throughout. Pass the same `--ddim` to `analyze`.

## Outputs

```
results/fixed_budget__<backend>/                 fixed budget
results/target_quality_fid<target>__<backend>/   target quality, one folder per target FID (e.g. target_quality_fid50__native)
                                                 with --ddim N, "_ddimN" is added before "__<backend>"
  runs/<model>/seed<k>/
    config.yaml           resolved config used
    run.json              summary: environment, capacity, progress, idle power, energy per phase, quality, QpJ, inference
    progress.csv          step, images_seen, epoch, train_time_s, lr, losses
    energy_segments.csv   one row per measured segment
    fid_history.csv       target_quality only
    inference.csv         J and seconds per 1,000 images per sampler
    samples.png           64 final samples
    model_final.pt        not tracked in git
    backend_raw/          the backend's own output
    stdout.log            not tracked in git
  summary/                written by `analyze`
    runs.csv              one row per run
    summary.csv           mean and std per model across seeds
    plots/                fid_vs_energy, energy_breakdown, cumulative_energy, inference_energy, qpj, fid_vs_epoch
```

**QpJ = 1 / (FID × E)**, where E is in joules, reported for three scopes: training (T), training + periodic FID checks (T+E), and training + final sampling (T+S). FID uses 10k generated images against cached statistics of 50k CelebA training images (Inception-v3, torch-fidelity weights). The quality target's periodic checks use 5k generated images.

## Recommended first session on a new machine

1. Run `bash setup.sh`.
2. In `results/setup_checks/*.txt`, confirm that `measured` is `True` for CPU and GPU.
3. Run `python run_all.py train --models dcgan --seeds 42 --set experiment.results_dir=results_timing --set stopping.fixed_budget.budget="{unit: epochs, value: 1}"` as a timing check. It writes to a separate folder, so the real results stay clean.
4. Launch the full suite inside `tmux` or `nohup`.