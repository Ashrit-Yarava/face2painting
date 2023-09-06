"""
File containing all the loss functions for the models used.
"""


import jax
import jax.numpy as jnp
import haiku as hk
import optax
from src.utils import Gan, ModelState
from src.generator import generator
from src.discriminator import discriminator


def criterion(predictions: jnp.array, target: jnp.array):
    """
    Criterion function used as the calculation for identifying
    difference between the predictions and the target.

    The generator and discriminator loss functions call on this. 

    @param predictions: The model outputs.
    @param targets: The goal model outputs.
    @return: Float value denoting the loss.
    """
    return jnp.mean(optax.l2_loss(predictions, target))


def discriminator_loss(a2b_params: hk.Params, a2b: Gan, b2a: Gan,
                       b_real: jnp.array, b_fake: jnp.array):
    """
    Calculate the discriminator loss using the criterion function
    on real predictions and fake predictions.

    Real predictions have a target value of 1 while fake predictions
    have a vlaue of 0.

    @param a2b_params: The parameters of the discriminator model.
    @param a2b: The GAN segment that the discriminator params belong to.
    @param b2a: The other GAN segment.
    @param b_real: The training input that the disciminator is trying to
                   identify.
    @param b_fake: The GAN generator output.
    @return: Float value of the combined loss.
    """
    real_preds = discriminator.apply(a2b_params, b_real)
    real_loss = criterion(real_preds, jnp.ones_like(real_preds))

    fake_preds = discriminator.apply(a2b_params, b_fake)
    fake_loss = criterion(fake_preds, jnp.ones_like(fake_preds))

    return (real_loss + fake_loss) * 0.5


def generator_loss(a2b_params: hk.Params, a2b: Gan, b2a: Gan,
                   a_real: jnp.array):
    """
    Generator loss calculate the discriminator loss and the cycle
    consistency loss.

    Discriminator loss involves identifying the error the
    discriminator makes by comparing it with the value of true output.

    Cycle consistency loss aims to calculate the similarity between the
    model inputs and the outputs of the other generator model.
    Ideally the models should have similar outputs.

    @param a2b_params: The Generator that converts from A to B.
    @param a2b: The GAN segment the generator params belong to.
    @param b2a: The other GAN segment.
    @param a_real: The inputs for the generator model.
    @return: The combined loss of the generator.
    """

    # Loss from discriminator
    b_fake = generator.apply(a2b_params, a_real)
    disc_output = discriminator.apply(a2b.discriminator.params, b_fake)
    disc_loss = criterion(disc_output, jnp.ones_like(disc_output))

    a_fake = generator.apply(b2a.generator.params, b_fake)
    cycle_loss = criterion(a_fake, a_real)  # Using mean reduction

    return (disc_loss + cycle_loss) * 0.5  # Take the mean of both the outputs.
