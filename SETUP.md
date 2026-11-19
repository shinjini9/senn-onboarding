# SENN Task 2: Memory Wrap Onboarding

**GitHub repo:** https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_NEW_REPO_NAME>

## Workstation setup
- **Platform:** Google Colab
- **GPU:** Tesla T4, 15360 MiB
- **RAM:** 12Gi
- **CPUs:** 2
- **OS:** Ubuntu 24.04.4 LTS
- **Python / PyTorch / torchvision / CUDA:** 3.13.16 / 2.14.1+cu130 / 0.29.1+cu130 / 13.0

## Additional steps I had to take
The repo targets Python 3.8 / PyTorch 1.9.1 through Docker or Conda. I used Google Colab instead:

1. **Environment:** used Colab's preinstalled PyTorch/torchvision and installed the rest with
   `pip install memorywrap entmax captum absl-py pyyaml`.
2. **SVHN download error:** torchvision's automatic download from the official server
   (`ufldl.stanford.edu/housenumbers`) failed with a connection error. Fix: downloaded `train_32x32.mat` and
   `test_32x32.mat` from the same official URL over HTTPS with `wget -c` and retries, placed them in `paper/datasets/`,
   and checked them against torchvision's MD5 checksums, so torchvision skips its own download. No research script was modified.
3. **Generating figures:** ran `python3 generate_memory_images.py --path_model=../models/2000.pt --dir_dataset=../datasets/`
   from `paper/scripts` (the scripts use relative paths). Stopped it after 305 figures (the task says it does not need to finish).
4. **Random selection:** the script's test loader is not shuffled, so I picked 10 of the generated figures with
   `random.sample` (seed = 2026) and numbered them 01–10.

Pretrained model: `paper/models/2000.pt`, i.e. MobileNet + Memory Wrap (`encoder_memory`) trained on 2000 SVHN samples.

## Explanation figures
Each figure shows 3 test inputs. Top row: input + prediction. Bottom row: memory samples with positive weight ("Used Samples").
Files are in `selected_explanations/`.

| # | File | True labels (L→R) | Predictions correct? | Comment |
|---|---|---|---|---|
| 01 | `01_fig182.png` | [1, 0, 2] | | |
| 02 | `02_fig491.png` | [3, 1, 3] | | |
| 03 | `03_fig773.png` | [2, 9, 1] | | |
| 04 | `04_fig788.png` | [1, 1, 0] | | |
| 05 | `05_fig158.png` | [2, 6, 4] | | |
| 06 | `06_fig344.png` | [4, 5, 6] | | |
| 07 | `07_fig854.png` | [1, 0, 1] | | |
| 08 | `08_fig647.png` | [3, 6, 3] | | |
| 09 | `09_fig881.png` | [1, 3, 1] | | |
| 10 | `10_fig842.png` | [7, 8, 2] | | |

**Overall observations:** _TODO: Are the explanations useful/effective? Do the used samples share the predicted digit?
Are they similar in colour/font/background? How many samples are used? What do the memory samples show when the model is wrong?_
