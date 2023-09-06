import jax
import jax.numpy as jnp
import jax.random as jr

import pickle
from PIL import Image
from glob import glob

from src.generator import generator
from src.discriminator import discriminator
from src.utils import generate_sampled_image, Dataset


CHECKPOINT_FILE = "./checkpoints/50000.pickle"
A_GLOB = "./data/faces/*.*"
B_GLOB = "./data/vangogh/*.*"


if __name__ == "__main__":

    checkpoint = pickle.load(open(CHECKPOINT_FILE, "rb"))
    a2b = checkpoint["a2b"]
    b2a = checkpoint["b2a"]

    dataset = Dataset(glob(A_GLOB)[:20], glob(B_GLOB)[:20], 0)
    a_real, b_real = dataset.sample(1)

    a_real = jnp.asarray(a_real)
    b_real = jnp.asarray(b_real)

    generator_apply = jax.jit(generator.apply)
    generated_image = generate_sampled_image(
        a2b, b2a, a_real, b_real, generator_apply)

    generated_image.save("./samples/sample.png")
