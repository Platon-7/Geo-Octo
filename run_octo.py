import jax
import numpy as np
from octo.model.octo_model import OctoModel

observation = {
    # 'image_primary' is the key for the main camera image.
    # The image must have a batch and history dimension, hence the [np.newaxis, np.newaxis, ...].
    # Shape becomes (1, 1, 256, 256, 3) --> (batch, history, height, width, channels).
    "image_primary": np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)[np.newaxis, np.newaxis, ...],

    # 'timestep_pad_mask' tells the model that this single timestep is a valid observation.
    # Shape is (1, 1) --> (batch, history).
    "timestep_pad_mask": np.array([[True]], dtype=bool),
}
print(f"Created observation with image of shape: {observation['image_primary'].shape}")

model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
task = model.create_tasks(texts=["pick up the spoon"])
action = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))

# 5. Print the result. The action will be a numerical array.
print("\nPredicted Action:")
print(action)
print("\nTest successful!")