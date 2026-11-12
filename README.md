# MatServe

MatServe studies retrieval, adapter refresh and foreground serving on a shared-memory edge GPU. The implementation includes context-budget calibration, coreset selection, branchwise preference training and serving from committed adapter snapshots.

```
src/          shared model, planning, training and serving code
experiments/  stream, executor, retrieval/coreset and headline experiments
analysis/     tables and figures from saved results
data/         prepared WildChat and LaMP-4 inputs
results/      completed experiments and raw power/event logs
```

## Setup

The experiments used Python 3.12, PyTorch 2.12.0+cu130, Transformers 5.11.0 and PEFT 0.19.1 on Jetson AGX Thor. Install the CUDA-enabled PyTorch build for the machine, then the Python dependencies:

```sh
python -m pip install -r requirements.txt
```

Falcon-H1 and GatedDeltaNet runs also use their model-specific Mamba/FLA and causal-convolution kernels.

Set `MATSERVE_MODEL_DIR` to the directory containing local model downloads:

```sh
export MATSERVE_MODEL_DIR=/path/to/models
```

| Directory | Model revision used |
|---|---|
| `Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| `Falcon-H1-1.5B-Deep-Instruct` | `b6648636ddc906688974282de6e7a243395f5423` |
| `Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

Model weights are downloaded separately. The 27B loader uses the Transformers `qwen3_5` text implementation.

## Analyze saved results

Run from the repository root:

```sh
python analysis/stream_executor.py
python analysis/controls.py
python analysis/calibration.py --hybrid-from-control
python analysis/headlines.py
python analysis/online.py
```

Tables, plots and JSON summaries are written to `outputs/`. These commands read the bundled results without loading a model. The calibration figure uses Dense/Mamba stream measurements and the 27B retrieval-control calibration.

## Run experiments

Use a new output path for each run. These examples use the Dense 1.5B backbone:

```sh
python experiments/run_replay.py --suite stream --backbone dense_qwen15b --data-dir data/wildchat_stream_20u --users 20 --turns 36 --out outputs/stream.json
python experiments/run_replay.py --suite executor --backbone dense_qwen15b --data-dir data/wildchat_stream_20u --users 1 --repetitions 4 --requests 48 --out outputs/executor.json
python experiments/run_controls.py --suite retrieval --backbone dense_qwen15b --data-dir data/wildchat_stream_20u --users 6 --out outputs/retrieval.json
python experiments/run_controls.py --suite coreset --backbone dense_qwen15b --data-dir data/wildchat_stream_20u --users 6 --out outputs/coreset.json
python experiments/run_headlines.py --backbone dense_qwen15b --lamp-dir data/lamp4_time_20u --authors 6 --max-wall-s 3600 --out outputs/headlines.json
python experiments/run_online.py --lamp-dir data/lamp4_time_20u --groups 6 --queries 24 --max-wall-s 2400 --out outputs/online.json
```

On the shared Thor machine, submit GPU commands through the existing serialized `qrun.py` queue. CPU checks use `CUDA_VISIBLE_DEVICES=""`. Keep paper compilation and result analysis on the Mac.

## Results

The repository contains ten completed primary runs: two stored-order stream replays, two executor comparisons, four retrieval/coreset controls, one Dense headline-transfer run and one six-group online replay. `results/` also contains the recorded power traces and online event journal. Historical method names remain in result keys.

Stream scores use response-order proxies and estimated prefill/refresh costs. Executor energy covers the full fixed-arrival replay. Retrieval scoring, headline generation and online oracle feedback are separate measurements. Intervals resample the recorded groups or scheduling repetitions; they do not turn repeated requests into independent users.

The source is organized from the cleaned experiment code. Historical results retain the original paths and source hashes. Failed configurations and unfinished experiments are excluded.

## Data

WildChat-derived bundles use the stored final-versus-initial response as a positional label. Contact details in four publication copies are redacted. Source: Zhao et al., *WildChat: 1M ChatGPT Interaction Logs in the Wild* (ICLR 2024), [dataset and ODC-BY terms](https://huggingface.co/datasets/allenai/WildChat).

LaMP-4 packets use the time-based benchmark from Salemi et al., *LaMP: When Large Language Models Meet Personalization*. The first twelve profile packets are included. See the [benchmark](https://lamp-benchmark.github.io/), [downloads](https://lamp-benchmark.github.io/download) and [upstream CC-BY-NC-SA notice](https://github.com/LaMP-Benchmark/LaMP/blob/main/CC-BY-NC-SA-4.0.txt).
