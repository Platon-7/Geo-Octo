from huggingface_hub import snapshot_download
import os

# The ID of the dataset repository on Hugging Face
repo_id = "openvla/modified_libero_rlds"

# The local directory where you want to save the dataset
destination_directory = "/scratch-shared/tmp.cwkV8vOvfY/libero_datasets"

print(f"Starting download of '{repo_id}'...")
print(f"This will save the dataset to: {destination_directory}")
print("This will take some time and download ~10 GB of data.")

# This command downloads the entire repository snapshot.
# It automatically handles LFS files without needing the git-lfs client.
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=destination_directory,
    local_dir_use_symlinks=False  # Good practice on shared systems
)

print("Download complete!")