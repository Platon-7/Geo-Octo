import os
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square


def extract_layer_features(images: torch.Tensor, model: VGGT):
	# Expect images: [S, 3, H, W]; ensure batch dim for aggregator
	if images.dim() == 4:
		images = images.unsqueeze(0)

	with torch.no_grad():
		aggregated_tokens_list, patch_start_idx = model.aggregator(images)

	# Collect only patch tokens from all layers
	feats = []
	for x in aggregated_tokens_list:  # each: [B, S, P, 2C]
		assert x.dim() == 4 and x.shape[0] == 1, "Expect B==1"
		x_tokens = x[0, :, patch_start_idx:, :]  # [S, Hp*Wp, 2C]
		S, P_sel, C2 = x_tokens.shape
		feats.append(x_tokens.reshape(S * P_sel, C2))  # [S*Hp*Wp, 2C]

	# Stack: [L, S*Hp*Wp, 2C] (with S=1 if a single image)
	return torch.stack(feats, dim=0), patch_start_idx


def main():
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# Default: single image; no manual paths needed
	image_names = [os.path.join("vggt", "examples", "kitchen", "images", f"{i:02}.png") for i in [0]]

	# Force 224x224 (divisible by patch size 14)
	target_size = 224
	images, _ = load_and_preprocess_images_square(image_names, target_size=target_size)
	images = images.to(device)  # [S, 3, 224, 224]

	# Model
	model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
	model.eval()

	# Patch grid size (Hp, Wp)
	S, _, H, W = images.shape
	Hp = H // model.aggregator.patch_size
	Wp = W // model.aggregator.patch_size

	# Features
	features, patch_start_idx = extract_layer_features(images, model)  # [L, S*Hp*Wp, 2048]
	L, N_tokens, D = features.shape

	print(f"images: {image_names}")
	print(f"S={S}, H={H}, W={W}, patch_size={model.aggregator.patch_size}, Hp={Hp}, Wp={Wp}")
	print(f"patch_start_idx={patch_start_idx}")
	print(f"features shape: {tuple(features.shape)}  (L, S*Hp*Wp, 2048)")
	if S == 1:
		print(f"single-image tokens (Hp*Wp): {Hp*Wp}")
		assert N_tokens == Hp * Wp == (target_size // 14) * (target_size // 14), "Unexpected token count"
	print(f"layers (L): {L}, dim: {D}")


if __name__ == "__main__":
	main()