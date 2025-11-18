Geo Octo Project Guide
======================

Environment Setup
-----------------

Run all steps from the repository root (`geo_octo`):

1. Create the Conda environment specified by the project:

   ```
   conda env create -f environment.yml
   ```

2. Install the shared Python dependencies that apply to both Octo and LIBERO components:

   ```
   pip install -r requirements_octo_libero.txt
   ```

3. Install the editable packages. Run each command from the repo root. If your environment cannot locate the modules afterwards, export `PYTHONPATH=$PYTHONPATH:/path/to/geo_octo` (plus the relevant subpackages).

   ```
   pip install -e octo
   pip install -e LIBERO
   ```

4. Install the CUDA-enabled JAX runtime:

   ```
   pip install --no-deps jaxlib==0.4.20+cuda11.cudnn86 \
     -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

5. Install the CUDA build tools required by VGGT/Octo and export the helper environment variables (consider appending the exports to your shell profile so they persist):

   ```
   pip install nvidia-cuda-nvcc-cu11==11.8.89 nvidia-cuda-nvrtc-cu11==11.8.89

   export CUDA_NVCC_BIN=$(python -c "import os,inspect,nvidia.cuda_nvcc as nvcc; \
       print(os.path.join(os.path.dirname(inspect.getfile(nvcc)),'bin'))")
   export NVRTC_LIB_DIR=$(python -c "import os,inspect,nvidia.cuda_nvrtc as nvrtc; \
       print(os.path.join(os.path.dirname(inspect.getfile(nvrtc)),'lib'))")
   export PATH="$CUDA_NVCC_BIN:$PATH"
   export LD_LIBRARY_PATH="$NVRTC_LIB_DIR:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
   ```

Finetuning Jobs
---------------

All submission scripts live under `snellius_jobs/` and automatically derive their paths from the original checkout directory (where you run `sbatch`). Use the following jobs for each finetuning variant:

- **Baseline Octo finetuning:** `snellius_jobs/finetune_octo_baseline.job`
- **VGGT fusion finetuning:** `snellius_jobs/finetune_vggt_online.job`
- **VGGT-only finetuning:** reuse `snellius_jobs/finetune_vggt_online.job` but ensure the job arguments include `--use_vision_encoder=False`.
- **VGGT pointmap finetuning:** `snellius_jobs/finetune_pointmap.job`

Evaluation Jobs
---------------

- **Baseline policy evaluation:** `snellius_jobs/evaluate_octo.job`
- **VGGT fusion evaluation:** `snellius_jobs/evaluate_octo_vggt_torch.job`
- **VGGT-only evaluation:** reuse `snellius_jobs/evaluate_octo_vggt_torch.job` and pass `--use_vision_encoder=False`.
- **VGGT pointmap evaluation:** `snellius_jobs/evaluate_octo_vggt_pointmap.job`
