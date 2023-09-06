"""
Data loading and checkpointing utilities.
"""

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
from glob import glob
from collections import namedtuple
from tqdm import tqdm


SIZE = 256


ModelState = namedtuple("ModelState", ("params", "opt_state"))
Gan = namedtuple("Gan", ("generator", "discriminator"))


class Dataset:

    def __init__(self, a_files, b_files, seed):

        self.a = a_files
        self.b = b_files

        self.a_np = np.empty((len(self.a), SIZE, SIZE, 3), np.uint8)
        self.b_np = np.empty((len(self.b), SIZE, SIZE, 3), np.uint8)

        for i, painting in enumerate(tqdm(self.a)):
            img = Image.open(painting).resize((SIZE, SIZE)).convert("RGB")
            self.a_np[i, ...] = np.array(img)

        for i, photo in enumerate(tqdm(self.b)):
            img = Image.open(photo).resize((SIZE, SIZE)).convert("RGB")
            self.b_np[i, ...] = np.array(img)

        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size):

        a_idx = self.rng.integers(
            0, self.a_np.shape[0], batch_size)
        b_idx = self.rng.integers(0, self.b_np.shape[0], batch_size)

        a = self.a_np[np.array(a_idx)]
        a = normalize_image(a)

        b = self.b_np[np.array(b_idx)]
        b = normalize_image(b)

        return a, b


def normalize_image(img: np.array):
    """
    Function called to scale a given numpy image from 0-255 to -1 - 1.

    @param img: The numpy array of an image to scale.
    """
    return (img / 127.5) - 1


def load_image(image_file: str):

    img: Image = Image.open(image_file)
    return img


def convert_to_image(jax_img: jnp.array):
    """
    Convert a JAX array input for model to a proper image.

    @param jax_img: The JAX image to convert range -1 to 1.
    @return: An image object.
    """
    np_img = np.asarray((jax_img + 1) * 127.5, dtype=np.uint8)
    return Image.fromarray(np_img[0])


def generate_sampled_image(a2b: Gan,
                           b2a: Gan,
                           a_real: jnp.array,
                           b_real: jnp.array,
                           generator_apply):
    """
    Generate a collaged image of the 2 input images and the
    generated image output and return the image.

    Arranged as:

    ┌───┬─────┐
    │ A │ A_g │
    ├───┼─────┤
    │ B │ B_g │
    └───┴─────┘

    Where _g represents the generated image.

    @param a2b: The First generator segment
    @param b2a: The second generator segment.
    @param a_real: The image to be passed into a2b.
    @param b_real: the image to be passed into b2a.
    @return: The PIL image of the 4 images.
    """

    b_fake = generator_apply(a2b.generator.params, a_real)
    a_fake = generator_apply(b2a.generator.params, b_real)

    a_real = convert_to_image(a_real)
    a_fake = convert_to_image(a_fake)
    b_real = convert_to_image(b_real)
    b_fake = convert_to_image(b_fake)

    final_image = Image.new("RGB", (256 * 2, 256 * 2))
    final_image.paste(a_real, (0, 0))
    final_image.paste(a_fake, (256, 0))
    final_image.paste(b_real, (0, 256))
    final_image.paste(b_fake, (256, 256))

    return final_image


def initialize_directories():
    checkpoint_dir = Path("./checkpoints/")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir()

    samples_dir = Path("./samples/")
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir()



if __name__ == "__main__":
    # Test that the function works.

    initialize_directories()
    
    painting_files = glob("./data/vangogh/*")[0:10]
    photos_files = glob("./data/faces/*")[0:10]

    dataset = Dataset(painting_files, photos_files, 0)

    painting, photo = dataset.sample(1)

    painting = convert_to_image(painting)
    photo = convert_to_image(photo)
