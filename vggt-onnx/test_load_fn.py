import os

from load_fn import load_and_preprocess_images
from vggt.utils.load_fn import load_and_preprocess_images as load_and_preprocess_images_original


def test_equal_arrays(image_names):
    images_torch = load_and_preprocess_images_original(image_names, "pad")
    images_numpy = load_and_preprocess_images(image_names)
    assert images_torch.shape == images_numpy.shape
    assert (images_torch.cpu().numpy() == images_numpy).all()


image_names_same_sizes = [
    os.path.join("vggt", "examples", "kitchen", "images", f"{i:02}.png")
    for i in range(5)
]
image_names_different_sizes = [
    os.path.join("vggt", "examples", "room", "images", f"no_overlap_{suffix}")
    for suffix in ["1.png", "2.jpg"]
]

test_equal_arrays(image_names_same_sizes[:1])
test_equal_arrays(image_names_same_sizes)
test_equal_arrays(image_names_different_sizes)
