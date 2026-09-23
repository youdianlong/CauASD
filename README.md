# CauASD

Official implementation of **Causal Action Speed Disentanglement for Zero-Shot Skeleton-Based Action Recognition**.

## Requirements

```bash
pip install -r requirements.txt
```

## Data

Place the processed skeleton data for NTU-60, NTU-120, and PKU-MMD, together with the language embeddings, under:

```text
data/zeroshot/<dataset>/split_<split>/
data/language/
```

The dataset and pretrained checkpoints are not included in this repository.
For NTU-RGB+D and PKU-MMD data download and preprocessing, please refer to the
[SMIE Data Preparation instructions](https://github.com/YujieOuO/SMIE#data-preparation).

## Training

Run the main training file from the project root:

```bash
python main.py
```

The default configuration is defined in `config.py`. Training checkpoints and logs are saved under `output/`.
To run another dataset or split, change `dataset` and `split` in `config.py` before training.

## Other experiments

The Shift-GCN setting uses the `sota` branch in `main.py`. Build the CUDA extension first if needed:

```bash
cd module/Temporal_shift
bash run.sh
cd ../..
python main.py with track="sota"
```

The temporal speed evaluations are provided in `experiments/`:

- `test_speed_warps.py`: uniformly warped and nonlinear temporally perturbed samples;
- `test_variable_length_speed.py`: variable-length speed evaluation;
- `natural_speed/`: natural-speed stratification and duration-aware sensitivity analysis.

## Author

Author: Dianlong Y and Anqi Wang  
Coder: Anqi Wang  
Email: [waq@stumail.ysu.edu.cn](mailto:waq@stumail.ysu.edu.cn)

This package was developed by Anqi Wang. For questions concerning the code, please contact Anqi Wang by email.
