# Quality per Joule v2

A controlled comparison of the energy efficiency of five generative model families (AE, VAE, DCGAN, WGAN-GP and DDPM) on CelebA at 32×32.

Every model uses the canonical architecture and hyperparameters of its original publication. Energy is measured per segment for CPU, RAM and GPU, using one of three interchangeable backends.

## Design

**Shared backbone.** The AE/VAE decoder and both GAN generators are the same Radford et al. (2016) generator. The AE/VAE encoder, the DCGAN discriminator and the WGAN-GP critic are the same mirrored encoder. Energy differences between these four models therefore reflect the training paradigm rather than network capacity.

| Model | Architecture | Objective | Optimizer | Reference |
|---|---|---|---|---|
| AE | Shared encoder/decoder, z = 128. Samples are drawn from a full-covariance Gaussian fitted to the training latents. | Summed squared error | Adam 2e-4, β (0.5, 0.999) | Hinton & Salakhutdinov 2006; Ghosh et al. 2020 |
| VAE | Shared encoder (μ, log σ²) and decoder | Gaussian NLL (σ² = 0.5) + KL, β = 1 | Adam 2e-4, β (0.5, 0.999) | Kingma & Welling 2014 |
| DCGAN | Shared generator and discriminator (BatchNorm) | Non-saturating BCE | Adam 2e-4, β (0.5, 0.999) | Radford et al. 2016 |
| WGAN-GP | Shared generator; critic uses LayerNorm | Wasserstein + GP, λ = 10, n_critic = 5 | Adam 1e-4, β (0, 0.9) | Gulrajani et al. 2017 |
| DDPM | U-Net with 35.7M parameters (ch 128, mult 1,2,2,2, attention at 16×16, dropout 0.1) | ε-MSE, T = 1000, linear β schedule | Adam 2e-4, 5k-step warmup, gradient clip 1, EMA 0.9999 | Ho et al. 2020 |

**Held constant across all models:** dataset and preprocessing (178-pixel center crop, resized to 32×32), batch size 64, FP32, seeds, hardware and the FID protocol.

**Epochs.** One epoch is one pass over the data for every model. WGAN-GP draws `n_critic` fresh batches per generator step.

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

`setup.sh` already runs `check` and `prepare`.

| Flag | Values | Default |
|---|---|---|
| `--backend` | `native`, `codecarbon`, `codegreen` | `energy.backend` (`native`) |
| `--stopping` | `fixed_budget`, `target_quality` | `stopping.mode` (`fixed_budget`) |
| `--target-fid` | number; `target_quality` only | `stopping.target_quality.target_fid` (25) |
| `--models` | any of `ae vae dcgan wgan_gp ddpm` | all five |
| `--seeds` | integers, space separated | `42 2026 1337` |
| `--config` | YAML file merged over `config/default.yaml` | none |
| `--set` | `key.path=value`, repeatable | none |
| `--overwrite` | re-run finished runs | off |

Each (model, seed) run executes in its own subprocess. Runs that already finished are skipped, so an interrupted suite can simply be restarted. Runs that failed or were interrupted are re-run from the start. Before any training starts, the suite checks every finished run's `config.yaml` against the current config. If a setting that affects results has changed, the suite stops and asks for `--overwrite` or a different `experiment.results_dir`.

```bash
python run_all.py train --backend codegreen --stopping target_quality --models dcgan ddpm --seeds 42
python run_all.py train --set stopping.fixed_budget.budget="{unit: energy_j, value: 2.0e6}"
python run_all.py train --config config/smoke.yaml --backend codecarbon     # quick CPU test, synthetic data
```

## Energy measurement

Only the selected backend runs. Every backend reports the same fields for each segment: `cpu_j`, `ram_j`, `gpu_j`, `total_j = cpu + ram + gpu`, `duration_s`, and `measured` flags that show which values were measured rather than estimated.

| | `native` (standard library only) | `codecarbon` | `codegreen` |
|---|---|---|---|
| CPU | RAPL `package-*` (sysfs) | RAPL if readable, else a TDP/load estimate | RAPL package |
| RAM | RAPL `dram`; estimated from `energy.ram_w_per_gb` if absent | Always its model estimate | RAPL `dram`; estimated if absent |
| GPU | NVML total-energy counter (ctypes) | NVML | NVML |

**Segments.** `idle` (baseline, used to compute `net_total_j`), `train_<k>`, `gaussian_fit` (AE only, counted as training), `eval_<k>` (target quality only), `final_<sampler>` and `inference_<sampler>`.

**Limitation.** Software counters cannot measure whole-system power at the wall socket. `total_j` covers CPU package, DRAM and GPU only.

## Stopping criteria

- **`fixed_budget`:** stop at `budget`, given in `epochs | steps | images | wall_clock_s | energy_j`. Training is measured in segments of size `segment`. Energy budgets are checked at segment ends, so a run can overshoot by at most one segment.
- **`target_quality`:** every `eval_every`, compute FID on `fid_samples` images. Stop when FID ≤ `target_fid` (status `target_reached`), or stop at `max_budget` (status `budget_exhausted`). DDPM's periodic checks use DDIM-100 as a proxy; its final FID always uses 1000-step ancestral sampling.

## References

PDFs of the papers behind each model and method are kept locally in `papers/`, which is not tracked in git.

## Outputs

```
results/fixed_budget__<backend>/                 fixed budget
results/target_quality_fid<target>__<backend>/   target quality, one folder per target FID (e.g. target_quality_fid30__native)
  runs/<model>/seed<k>/
    config.yaml           resolved config used
    run.json              summary: environment, capacity, progress, idle power, energy per phase, quality, QpJ, inference
    progress.csv          step, images_seen, epoch, train_time_s, lr, losses
    energy_segments.csv   one row per measured segment
    fid_history.csv       target_quality only
    inference.csv         J and seconds per 1,000 images per sampler
    samples.png           64 final samples
    model_final.pt
    backend_raw/          the backend's own output
    stdout.log
  summary/
    runs.csv              one row per run
    summary.csv           mean and std per model across seeds
    plots/                fid_vs_energy, energy_breakdown, cumulative_energy, inference_energy, qpj, fid_vs_epoch
```

**QpJ = 1 / (FID × E)**, where E is in joules, reported for three scopes: training, training + evaluation, and training + final sampling. FID uses 10k generated images against cached statistics of 50k CelebA training images (Inception-v3, torch-fidelity weights).

## Recommended first session on a new machine

1. Run `bash setup.sh`.
2. In `results/setup_checks/*.txt`, confirm that `measured` is `True` for CPU and GPU.
3. Run `python run_all.py train --models dcgan --seeds 42 --set experiment.results_dir=results_timing --set stopping.fixed_budget.budget="{unit: epochs, value: 1}"` as a timing check. It writes to a separate folder, so the real results stay clean.
4. Launch the full suite inside `tmux` or `nohup`.
