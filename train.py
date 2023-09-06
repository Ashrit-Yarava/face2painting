"""
Training script for CycleGAN
----------------------------
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import haiku as hk
import optax

import numpy as np
from glob import glob
import logging
from datetime import datetime
import pickle

from src.generator import generator
from src.discriminator import discriminator
from src.utils import ModelState, Gan, Dataset, generate_sampled_image
from src.loss_functions import generator_loss, discriminator_loss


@jax.jit
def train_step(a2b: Gan,
               b2a: Gan,
               a_real: jnp.array,
               b_real: jnp.array):

    # Generator Loss

    # Train Generator A2B
    a2b_gen_loss, grads = generator_loss(a2b.generator.params, a2b, b2a, a_real)
    updates, opt_state = optimizer.update(
        grads, a2b.generator.opt_state, a2b.generator.params)
    params = optax.apply_updates(a2b.generator.params, updates)
    a2b = Gan(ModelState(params, opt_state), a2b.discriminator)

    # Train Generator B2A
    b2a_gen_loss, grads = generator_loss(b2a.generator.params, b2a, a2b, b_real)
    updates, opt_state = optimizer.update(
        grads, b2a.generator.opt_state, b2a.generator.params)
    params = optax.apply_updates(b2a.generator.params, updates)
    b2a = Gan(ModelState(params, opt_state), b2a.discriminator)

    # Discriminator Loss

    # Generate the fake images.
    a_fake = generator.apply(b2a.generator.params, b_real)
    b_fake = generator.apply(a2b.generator.params, a_real)

    # Train Discriminator A2B
    a2b_disc_loss, grads = discriminator_loss(a2b.discriminator.params, a2b, b2a, b_real, b_fake)
    updates, opt_state = optimizer.update(grads, a2b.discriminator.opt_state, a2b.discriminator.params)
    params = optax.apply_updates(a2b.discriminator.params, updates)
    a2b = Gan(a2b.generator, ModelState(params, opt_state))

    # Train Discriminator B2A
    b2a_disc_loss, grads = discriminator_loss(b2a.discriminator.params, b2a, a2b, a_real, a_fake)
    updates, opt_state = optimizer.update(grads, b2a.discriminator.opt_state, b2a.discriminator.params)
    params = optax.apply_updates(b2a.discriminator.params, updates)
    b2a = Gan(b2a.generator, ModelState(params, opt_state))

    return (a2b_gen_loss, b2a_gen_loss, a2b_disc_loss, b2a_disc_loss), a2b, b2a


if __name__ == "__main__":

    config = {
        "a_glob": "./data/landscapes/*.*",
        "b_glob": "./data/vangogh/*.*",
        "iterations": 1000000,
        "checkpoint_interval": 50000,
        "sampling_interval": 1000,
        "logging_interval": 500,
        "seed": 0
    }

    if config["seed"] is None:
        random_seed = int(np.random.random_integers(low=0, high=None, size=()))
        config["seed"] = random_seed

    logger = logging.basicConfig(level=logging.INFO, filename="output.txt", filemode="w")
    
    generator_loss = jax.value_and_grad(generator_loss)
    discriminator_loss = jax.value_and_grad(discriminator_loss)

    # Define the optimizer
    LEARNING_RATE = 1e-4
    optimizer = optax.adam(learning_rate=LEARNING_RATE)

    rng = hk.PRNGSequence(config["seed"])
    random_input = jr.uniform(next(rng), (1, 256, 256, 3), minval=-1, maxval=1)

    # Initialize A2B
    gen_params = generator.init(next(rng), random_input)
    gen_opt_state = optimizer.init(gen_params)

    disc_params = discriminator.init(next(rng), random_input)
    disc_opt_state = optimizer.init(disc_params)

    a2b = Gan(ModelState(gen_params, gen_opt_state), ModelState(disc_params, disc_opt_state))

    # Initialize A2B
    gen_params = generator.init(next(rng), random_input)
    gen_opt_state = optimizer.init(gen_params)

    disc_params = discriminator.init(next(rng), random_input)
    disc_opt_state = optimizer.init(disc_params)

    b2a = Gan(ModelState(gen_params, gen_opt_state), ModelState(disc_params, disc_opt_state))

    generator_apply = jax.jit(generator.apply)
    
    print("Initialized models.")

    dataset = Dataset(glob(config["painting_glob"])[:10], glob(config["photos_glob"])[:10], config["seed"])

    print("Loaded dataset.")

    losses, a2b, b2a = train_step(a2b, b2a, random_input, random_input)

    print("Compiled training step.")

    for index in range(config["iterations"]):

        if index % config["logging_interval"] == 0:
            start_time = datetime.now()

        photos, paintings = dataset.sample(1)
        photos = jnp.asarray(photos)
        paintings = jnp.asarray(paintings)
        losses, a2b, b2a = train_step(a2b, b2a, photos, paintings)

        if index % config["logging_interval"] == 0:
            end_time = datetime.now()

            a2b_log = f"A2B => G: {round(losses[0], 4)}\tD: {round(losses[2], 4)}"
            b2a_log = f"B2A => G: {round(losses[1], 4)}\tD: {round(losses[3], 4)}"

            logging.info(f"Iteration: {index} - {(end_time - start_time).total_seconds()}\n{a2b_log}\n{b2a_log}")

        if index % config["sampling_interval"] == 0 and index != 0:
            sample_image = generate_sampled_image(a2b, b2a, photos, paintings, generator_apply)
            sample_image.save(f"./samples/{index}.png")

        if index % config["checkpoint_interval"] == 0 and index != 0:
            checkpoint = { "a2b": a2b, "b2a": b2a }
            pickle.dump(checkpoint, open(f"./checkpoints/{index}.pickle", "wb"))

    checkpoint = { "a2b": a2b, "b2a": b2a }
    pickle.dump(checkpoint, open(f"./checkpoints/{index}-final.pickle", "wb"))