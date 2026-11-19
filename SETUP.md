# SENN Task 2: Memory Wrap Onboarding

**GitHub repo:** https://github.com/shinjini9/senn-onboarding

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

| # | File | True labels (L→R) | Predicted (L→R) | Correct? | Comment |
|---|---|---|---|---|---|
| 01 | `01_fig182.png` | [1, 0, 2] | [1, 8, 2] | 2/3 | The middle "0" is very blurry and was predicted as 8. Its used samples mix 8s with 0s/10s, so the explanation shows the 0-vs-8 confusion (both are closed loops). The 1 and the 2 have clean sets (almost all 1s, almost all 2s). |
| 02 | `02_fig491.png` | [3, 1, 3] | [3, 1, 3] | 3/3 | All correct. The 1 has a clean set of 1-containing numbers. Both 3s are blurry and get noisier sets (3s plus 7, 8/9 and 4), and they share many of the same samples because inputs in one figure use the same memory batch. |
| 03 | `03_fig773.png` | [2, 9, 1] | [2, 9, 1] | 3/3 | The clearest figure: every used sample shows the predicted digit (2s, 9s, 1s), across many colours and fonts. A convincing explanation. |
| 04 | `04_fig788.png` | [1, 1, 0] | [1, 1, 0] | 3/3 | All correct, with clean, high-contrast inputs. The two 1s get nearly identical sets of 1s; the 0 gets 0s/10s/70. The samples match the digit's shape, not the input's white/blue colours. |
| 05 | `05_fig158.png` | [2, 6, 4] | [8, 5, 4] | 1/3 | Two errors. The dark, blurry "2" was predicted as 8, and every used sample is an 8. The clearly visible "6" was predicted as 5, and its used samples are almost all 5s with no 6 at all. The explanation shows the model's reasoning (it matched the 6 to curvy 5s), not the true label, which makes the error easy to diagnose. The 4 is correct, with 4s. |
| 06 | `06_fig344.png` | [4, 5, 6] | [4, 5, 6] | 3/3 | All correct. The bright "4" uses only 8 samples, all 4s. The dark, low-contrast 5 and 6 use many more samples (up to 20) mixing 5, 6, 3 and 8, so the model looks less certain even though it's right. |
| 07 | `07_fig854.png` | [1, 0, 1] | [1, 0, 1] | 3/3 | All correct. The 1s get mostly 1-containing numbers (13, 14, 18, 19). The washed-out 0 gets a mixed set (0, 10, 60 but also 27, 2, 12). The last input is almost black and the digit is barely visible, yet the prediction is still correct. |
| 08 | `08_fig647.png` | [3, 6, 3] | [3, 6, 3] | 3/3 | All correct. The sharp 3 uses mostly 3s; the 6 uses 6s plus some 8s (58, 98). The blurry 3 on the right adds 8s and 88, showing the 3/8 similarity. |
| 09 | `09_fig881.png` | [1, 3, 1] | [1, 3, 1] | 3/3 | All correct. The 1s get many 1-containing samples (11, 21, 31). The ambiguous middle "3" uses a small set mixing 3s with 7s (7, 17), which shows the model hesitating between 3 and 7. |
| 10 | `10_fig842.png` | [7, 8, 2] | [7, 8, 2] | 3/3 | All correct. Each set is dominated by the predicted digit (7s; 8s and 88; 2s), with just a few look-alikes (3 in the 7 set, 3/29 in the 8 set, 4/94 in the 2 set). |

**Overall:** 27 of 30 inputs (90%) were classified correctly.

**Overall observations:**
- **The explanations are generally useful.** In most cases the used samples clearly contain the predicted digit, so you can see at a glance why the model chose that class.
- **They explain the model's decision, not the ground truth.** In the 3 errors (fig 01 middle, fig 05 left and middle), the used samples belong to the wrong predicted class (8s, 5s). That makes errors easy to diagnose: the 6→5 mistake in fig 05 is explained by the model matching the input to curvy 5s.
- **The used samples work as an uncertainty signal.** Clear, high-contrast inputs get a small, consistent set (often 8–11 samples of one digit). Blurry or dark inputs get larger, mixed sets (up to 20 samples) that also include look-alike digits (3/8, 3/7, 0/8, 5/6).
- **Matching is based on digit shape, not colour.** Samples vary widely in background and font colour compared with the input.
- **Limitations:** SVHN crops often contain neighbouring digits (e.g. "31" used to explain a 1), so it isn't always obvious which digit in a sample mattered. Inputs in the same figure share one memory batch, so they often reuse the same samples.
