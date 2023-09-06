import jax
import jax.numpy as jnp
import jax.random as jrnd
import haiku as hk


def leaky_relu(x):
    return jax.nn.leaky_relu(x, 0.2)


class Discriminator(hk.Module):
    """
    For discriminator net- works, we use 70 × 70 PatchGAN [22]. Let Ck
    denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k
    filters and stride 2. After the last layer, we apply a convo-
    lution to produce a 1-dimensional output. We do not use
    InstanceNorm for the first C64 layer. We use leaky ReLUs with a
    slope of 0.2. The discriminator architecture is: C64-C128-C256-C512
    """

    def __init__(self):
        """
        Initialize the PatchGAN discriminator.
        """
        super(Discriminator, self).__init__()

        self.layers = [

            hk.Conv2D(output_channels=64, kernel_shape=4,
                      stride=2, padding="VALID"),
            leaky_relu,

            hk.Conv2D(output_channels=128, kernel_shape=4,
                      stride=2, padding="VALID"),
            hk.InstanceNorm(True, True),
            leaky_relu,

            hk.Conv2D(output_channels=256, kernel_shape=4,
                      stride=2, padding="VALID"),
            hk.InstanceNorm(True, True),
            leaky_relu,

            hk.Conv2D(output_channels=512, kernel_shape=4,
                      stride=2, padding="VALID"),
            hk.InstanceNorm(True, True),
            leaky_relu,

            hk.Conv2D(output_channels=1, kernel_shape=14,
                      stride=1, padding="VALID"),
            jax.nn.sigmoid
        ]

        self.layers = hk.Sequential(self.layers)

    def __call__(self, x):
        """
        Call the PatchGAN discriminator.

        @param x: The jax array input image (batch size, 256, 256, 3)
        @return: A (batch size, 1) array in the range of 0 - 1.
        """
        x = self.layers(x)
        x = x.squeeze(-1).squeeze(-1)
        return x


@hk.without_apply_rng
@hk.transform
def discriminator(x):
    x = Discriminator()(x)
    return x


if __name__ == "__main__":

    key = hk.PRNGSequence(0)

    random_input = jrnd.uniform(next(key), (1, 256, 256, 3))
    params = discriminator.init(next(key), random_input)

    sample_output = discriminator.apply(params, random_input)
    print("Output Shape:", sample_output.shape)
