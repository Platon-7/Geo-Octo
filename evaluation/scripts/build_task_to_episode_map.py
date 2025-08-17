import os
import json
import argparse
import tensorflow as tf
from typing import Optional
import re


def extract_language(parsed):
	"""Try multiple keys and formats to robustly extract a language string from a parsed Example."""
	candidates = []
	# FixedLen string candidates
	for key in [
		'observation/language_instruction',
		'episode_metadata/task_description',
		'language_instruction',
		'task_description',
		'task_name',
	]:
		v = parsed.get(key, None)
		if v is not None:
			try:
				val = v.numpy().decode('utf-8') if isinstance(v.numpy(), (bytes, bytearray)) else str(v.numpy())
				candidates.append(val)
			except Exception:
				pass
	# VarLen string candidates (take first non-empty)
	for key in [
		'steps/observation/language_instruction',
		'steps/task/language',
		'steps/language_instruction',
	]:
		v = parsed.get(key, None)
		if v is not None and hasattr(v, 'values'):
			vals = v.values
			try:
				if tf.size(vals) > 0:
					val0 = vals[0].numpy().decode('utf-8')
					candidates.append(val0)
			except Exception:
				pass
	# Return first non-empty, stripped
	for c in candidates:
		c = (c or '').strip()
		if c:
			return c
	return ""


def extract_language_from_raw(raw_bytes: bytes) -> str:
	"""Parse tf.train.Example and heuristically pick language-like string fields."""
	ex = tf.train.Example()
	ex.ParseFromString(raw_bytes)
	keys = list(ex.features.feature.keys())
	# Collect candidate (key, str)
	kv = []
	for k in keys:
		feat = ex.features.feature[k]
		if feat.bytes_list.value:
			try:
				val = feat.bytes_list.value[0].decode('utf-8')
				if val:
					kv.append((k, val))
			except Exception:
				pass
	# Prefer keys with language/instruction
	patterns = [r'language', r'instruction', r'task_description', r'task_name']
	for pat in patterns:
		for k, v in kv:
			if re.search(pat, k, re.IGNORECASE):
				vs = v.strip()
				if vs:
					return vs
	# Fallback: the longest non-empty string
	if kv:
		k, v = max(kv, key=lambda x: len(x[1]))
		return v.strip()
	return ""


def build_mapping(dataset_dir: str, max_episodes: Optional[int] = None, inspect_first: int = 0):
	# Collect all TFRecord files
	tfrecord_files = tf.io.gfile.glob(os.path.join(dataset_dir, "*tfrecord*"))
	if not tfrecord_files:
		raise FileNotFoundError(f"No .tfrecord files found in '{dataset_dir}'.")
	# Dataset over all files
	ds = tf.data.TFRecordDataset(tfrecord_files)
	feature_description = {
		# Common image key (not used for mapping)
		'steps/observation/image': tf.io.VarLenFeature(tf.string),
		# Language keys
		'observation/language_instruction': tf.io.FixedLenFeature([], tf.string, default_value=b""),
		'episode_metadata/task_description': tf.io.FixedLenFeature([], tf.string, default_value=b""),
		'language_instruction': tf.io.FixedLenFeature([], tf.string, default_value=b""),
		'steps/observation/language_instruction': tf.io.VarLenFeature(tf.string),
		'steps/task/language': tf.io.VarLenFeature(tf.string),
		'task_description': tf.io.FixedLenFeature([], tf.string, default_value=b""),
		'task_name': tf.io.FixedLenFeature([], tf.string, default_value=b""),
	}
	by_language: dict[str, list[int]] = {}
	episodes: list[dict] = []
	for idx, raw in enumerate(ds):
		if max_episodes is not None and idx >= max_episodes:
			break
		if inspect_first and idx < inspect_first:
			ex = tf.train.Example.FromString(raw.numpy())
			print(f"[INSPECT] Episode {idx}: keys={list(ex.features.feature.keys())}")
			# Print sample values for keys containing language/instruction/task
			for k in ex.features.feature.keys():
				if re.search(r'(language|instruction|task)', k, re.IGNORECASE):
					feat = ex.features.feature[k]
					if feat.bytes_list.value:
						try:
							val = feat.bytes_list.value[0].decode('utf-8')
							print(f"  [INSPECT] {k} -> {val[:200]!r}")
						except Exception:
							pass
		parsed = tf.io.parse_single_example(raw, feature_description)
		lang = extract_language(parsed)
		if not lang:
			lang = extract_language_from_raw(raw.numpy())
		lang_norm = lang.strip()
		if lang_norm not in by_language:
			by_language[lang_norm] = []
		by_language[lang_norm].append(idx)
		episodes.append({"index": idx, "language": lang_norm})
		if (idx + 1) % 100 == 0:
			print(f"[INFO] Scanned {idx+1} episodes...")
	return {
		"dataset_dir": dataset_dir,
		"num_episodes_scanned": len(episodes),
		"by_language": by_language,
		"episodes": episodes,
	}


def main():
	parser = argparse.ArgumentParser(description="Build mapping from task language to episode indices by parsing TFRecords.")
	parser.add_argument("--dataset_dir", required=True, help="Path to dataset version dir, e.g., .../libero_spatial_no_noops/1.0.0")
	parser.add_argument("--out", required=True, help="Output JSON path for the mapping")
	parser.add_argument("--max_episodes", type=int, default=None, help="Optional cap on episodes to scan")
	parser.add_argument("--inspect_first", type=int, default=0, help="Print keys and sample values for the first N episodes")
	args = parser.parse_args()
	mapping = build_mapping(args.dataset_dir, args.max_episodes, args.inspect_first)
	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, 'w') as f:
		json.dump(mapping, f, indent=2)
	print(f"[SUCCESS] Wrote mapping to {args.out}")


if __name__ == "__main__":
	# ensure TF runs on CPU for parsing
	try:
		import tensorflow as tf  # noqa: F401
		tf.config.set_visible_devices([], 'GPU')
	except Exception:
		pass
	main()