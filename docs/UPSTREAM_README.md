# GATraj: A Graph- and Attention-based Multi-Agent Trajectory Prediction Model
Code for ["GATraj: A Graph- and Attention-based Multi-Agent Trajectory Prediction Model"](https://www.sciencedirect.com/science/article/pii/S092427162300268X)

![](imgs/introduction.gif)

# Environment
```
pip install -r requirements.txt
```

### Google Colab / Kaggle

The repository implements the **ETH/UCY pedestrian experiments**. It does not
include the nuScenes data adapter/evaluator used for the vehicle experiments
in the paper. The default data loader expects to be run from this directory
and reads the five benchmark scene CSVs from `data/`.

The GitHub clone includes ETH, Hotel, Zara1, Zara2 and the UCY University
benchmark CSVs. The separate `students003`, `uni_examples` and `zara03`
folders are auxiliary scene fragments and are not used in the standard
five-dataset leave-one-out evaluation. Dataset IDs are 0=ETH, 1=Hotel,
2=Zara1, 3=Zara2, 4=University.

For Colab, enable a GPU under **Runtime → Change runtime type**. For Kaggle,
enable **Settings → Accelerator → GPU**. Clone the repository in a fresh
notebook, install the requirements, then run from the `GATraj` directory:

```bash
git clone https://github.com/mengmengliu1998/GATraj.git
cd GATraj
python -m pip install -r requirements.txt
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))
PY
```

Train one fold (for example, hold out ETH):

```bash
python train.py --test_set 0 --num_epochs 1000 --x_encoder_layers 3 \
  --eta_min 1e-5 --batch_size 32 --learning_rate 5e-4 \
  --randomRotate True --final_mode 20 --neighbor_thred 10 \
  --using_cuda True --gpu 0 --clip 1 --pass_time 2 \
  --ifGaussian False --SR True --input_offset True
```

For a quick smoke run, set `--num_epochs 1` (this is only a runtime check, not
a meaningful reproduction). Colab/Kaggle sessions can disconnect. To train in
100-epoch sessions while keeping one 1,000-epoch cosine learning-rate schedule,
keep `--num_epochs 1000` fixed and set `--epochs_per_run 100`. Resume each
session with `--load_model <last completed epoch>` (use `0` for the first
session). Checkpoints include model, optimizer, scheduler and random-number
generator states. For example, in a Colab notebook cell, run:

```python
from pathlib import Path
import subprocess

epochs = range(100, 1001, 100)
for end_epoch in epochs:
    previous_epoch = end_epoch - 100
    command = [
        "python", "train.py",
        "--test_set", "1",
        "--num_epochs", "1000",
        "--epochs_per_run", "100",
        "--x_encoder_layers", "3",
        "--eta_min", "1e-5",
        "--batch_size", "32",
        "--learning_rate", "5e-4",
        "--randomRotate", "True",
        "--final_mode", "20",
        "--neighbor_thred", "10",
        "--using_cuda", "True",
        "--gpu", "0",
        "--clip", "1",
        "--pass_time", "2",
        "--ifGaussian", "False",
        "--SR", "True",
        "--input_offset", "True",
        "--seed", "1",
        "--load_model", str(previous_epoch),
    ]
    subprocess.run(command, check=True)

    # Mount Google Drive first; this preserves the checkpoints if Colab resets.
    source = Path("savedata/1/GATraj")
    destination = Path("/content/drive/MyDrive/GATraj/savedata/1/GATraj")
    destination.mkdir(parents=True, exist_ok=True)
    for checkpoint in source.glob("GATraj_*.tar"):
        target = destination / checkpoint.name
        if not target.exists() or checkpoint.stat().st_mtime > target.stat().st_mtime:
            target.write_bytes(checkpoint.read_bytes())
```

If a session stops unexpectedly, restore `savedata/1/GATraj/GATraj_<epoch>.tar`
from Drive and resume with `--load_model <epoch>`; retain the same seed and
training arguments. To test a checkpoint, add `--phase test --load_model
<epoch>`. Run IDs 0 through 4 for the five folds. Checkpoints made before
scheduler state was saved can be loaded, but their exact learning-rate history
cannot be reconstructed; for a strict reproducible comparison, start a fresh
run with this version.

### Dataset sources

- ETH: ETH/Hotel benchmark data and project resources are available from the
  [ETH Zurich Social Vision Group](https://www.vision.ee.ethz.ch/en/datasets/).
- UCY: the original UCY pedestrian benchmark is distributed with the
  [UCY Computer Vision datasets](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data/).
- The repository's `data/` files are already preprocessed `true_pos_.csv`
  files. If downloading raw benchmark data instead, convert it to the layout
  and CSV format expected by `utils.py`; these files are not plug-compatible
  with arbitrary ETH/UCY downloads without preprocessing.
- nuScenes is a separate, large autonomous-driving dataset. Access requires
  accepting its terms on the [official nuScenes download page](https://www.nuscenes.org/download).
  This repository does not currently provide the nuScenes training/evaluation
  pipeline, so the ETH/UCY commands above do not reproduce those results.

## Train
The Default settings are to train on ETH-eth. Data cache and models will be in the subdirectory "./savedata/0/".

```
git clone https://github.com/mengmengliu1998/GATraj.git
cd GATraj
python train.py --test_set <dataset to train> --num_epochs 1000 --x_encoder_layers 3 --eta_min 1e-5  --batch_size 32\
  --learning_rate 5e-4  --randomRotate True --final_mode 20 --neighbor_thred 10\
  --using_cuda True --clip 1 --pass_time 2 --ifGaussian False --SR True --input_offset True
```

Configuration files are also created after the first run, arguments could be modified through configuration files or command line.
Priority: command line \> configuration files \> default values in script.


The datasets are selected on arguments '--test_set'. Five datasets in ETH/UCY are corresponding to the value of \[0,1,2,3,4\] ([**eth, hotel, zara1, zara2, univ**]).

### Example

This command is to train model for ETH-eth
```
python train.py --test_set 0 --num_epochs 1000 --x_encoder_layers 3 --eta_min 1e-5  --batch_size 32\
  --learning_rate 5e-4  --randomRotate True --final_mode 20 --neighbor_thred 10\
  --using_cuda True --clip 1 --pass_time 2 --ifGaussian False --SR True --input_offset True
```

## Test
We provide the trained model weights in the subdirectory "./savedata/".
This command is to test model for ETH-eth, just add --phase test --load_model 1000 to the end of this training command.
```
python train.py --test_set 0 --num_epochs 1000 --x_encoder_layers 3 --eta_min 1e-5  --batch_size 32\
  --learning_rate 5e-4  --randomRotate True --final_mode 20 --neighbor_thred 10\
  --using_cuda True --clip 1 --pass_time 2 --ifGaussian False --SR True --input_offset True --phase test  --load_model 1000
```

### Cite GATraj

If you find this repo useful, please consider citing our paper
```bibtex
@article{cheng2023gatraj,
  title={Gatraj: A graph-and attention-based multi-agent trajectory prediction model},
  author={Cheng, Hao and Liu, Mengmeng and Chen, Lin and Broszio, Hellward and Sester, Monika and Yang, Michael Ying},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={205},
  pages={163--175},
  year={2023},
  publisher={Elsevier}
}
```

### Reference

The code base of dataloader is heavily adopted from [SR-LSTM](https://github.com/zhangpur/SR-LSTM). The visulization code base for nuScenes is adopted from [PGP](https://github.com/nachiket92/PGP).
