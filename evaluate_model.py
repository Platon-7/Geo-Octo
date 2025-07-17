import sys
from functools import partial
import jax
import numpy as np
import wandb
import gym
import tensorflow as tf
import os
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark

# Import necessary components from Octo and VGGT
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, RHCWrapper
from octo.utils.train_callbacks import supply_rng
from vggt.models.vggt import VGGT
# ===================================================================


# ===================================================================
# This is the VGGT Bridge, copied directly from your finetune.py
# ===================================================================
VGGT_MODEL_STATE = {"model": None, "device": None, "dtype": None, "captured_output": None}

def run_vggt_inference(image_tensor: tf.Tensor) -> np.ndarray:
    # ... (paste the entire, working run_vggt_inference function here) ...
    # It's identical to the one in your finetune.py
    # Lazy loads the model, uses the hook, etc.
    if VGGT_MODEL_STATE["model"] is None:
        print("Loading VGGT model for evaluation...")
        # ... rest of the function
    # ...
    return vggt_tokens.to(torch.float32).cpu().numpy()


def enrich_single_observation(observation: dict) -> dict:
    """Takes a SINGLE observation from a live rollout and adds VGGT tokens."""
    image_3d = observation["image_primary"]
    image_5d = np.expand_dims(image_3d, axis=(0, 1))
    vggt_tokens_5d = run_vggt_inference(tf.convert_to_tensor(image_5d))
    vggt_tokens_3d = np.squeeze(vggt_tokens_5d, axis=(0, 1))
    observation["vggt_tokens"] = vggt_tokens_3d
    return observation
# ===================================================================


def main():
    # This will be the path to the model you just saved (e.g., "./my_vggt_model_checkpoint/10")
    FINETUNED_PATH = "/gpfs/home4/pkarageorgis/geo_octo/octo/my_octo_vggt_model/octo_finetune/experiment_20250705_213435/10"
    
    wandb.init()

    print("Loading finetuned model...")
    model = OctoModel.load_pretrained(FINETUNED_PATH)

    # Use a valid LIBERO environment name
    env = gym.make("libero-90") 
    
    # Add wrappers for history and action chunking
    # The history horizon should match what your model was trained on
    env = HistoryWrapper(env, horizon=model.config['window_size'])
    env = RHCWrapper(env, exec_horizon=1) # Standard for Octo

    # Create the base policy function
    base_policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        ),
    )

    # Wrap the policy function with our VGGT enricher
    def wrapped_policy_fn(observation, tasks):
        enriched_observation = enrich_single_observation(observation)
        return base_policy_fn(enriched_observation, tasks)
    
    policy_fn = wrapped_policy_fn
    
    print("Starting rollouts...")
    for i in range(5): # Run 5 episodes
        obs, info = env.reset()
        # The language instruction might be empty, which is fine
        language_instruction = env.get_task().get("language_instruction", "")
        task = model.create_tasks(texts=[language_instruction])

        images = [obs["image_primary"][0]]
        episode_return = 0.0
        
        # Run rollout for a maximum of 500 steps
        for _ in range(500):
            # Add a batch dimension for the model
            model_obs = jax.tree_map(lambda x: x[None], obs)
            actions = policy_fn(model_obs, task)
            actions = actions[0] # Remove the batch dimension

            obs, reward, done, trunc, info = env.step(actions)
            
            # Log the primary image from the observation
            images.append(obs["image_primary"][0])
            episode_return += reward
            if done or trunc:
                break
        
        print(f"Episode {i+1} finished with return: {episode_return}")
        # Log the video of the rollout to wandb
        wandb.log({f"rollout_video_{i}": wandb.Video(np.array(images).transpose(0, 3, 1, 2), fps=20)})

if __name__ == "__main__":
    # You'll need to manually define the run_vggt_inference function above.
    # This is a simplified main execution for a standalone script.
    # For a more robust script, consider using absl.app like the original.
    main()