import jax
import jax.numpy as jnp
import jax.random as jr
import haiku as hk

from src.resnet_block import ResnetBlock




class Generator(hk.Module):
    """
    Generator architectures 

    We adopt our architectures from Johnson et
    al. [23]. We use 6 residual blocks for 128 × 128 training images,
    and 9 residual blocks for 256 × 256 or higher-resolution training
    images. Below, we follow the naming convention used in the Johnson
    et al.’s Github repository.

    Let c7s1-k denote a 7 × 7 Convolution-InstanceNorm- ReLU layer with k
    filters and stride 1. dk denotes a 3 × 3 Convolution-InstanceNorm-
    ReLU layer with k filters and stride 2. Reflection padding was
    used to reduce artifacts. Rk denotes a residual block that
    contains two 3 × 3 con- volutional layers with the same number of
    filters on both layer. uk denotes a 3 × 3 fractional-strided-
    Convolution- InstanceNorm-ReLU layer with k filters and stride 12 .

    The network with 6 residual blocks consists of:
    c7s1-64,d128,d256,R256,R256,R256,
    R256,R256,R256,u128,u64,c7s1-3

    The network with 9 residual blocks consists of:
    c7s1-64,d128,d256,R256,R256,R256,
    R256,R256,R256,R256,R256,R256,u128
    u64,c7s1-3
    """

    def __init__(self):
        """
        Initialize the model layers for the 9 block generator model.
        """
        super(Generator, self).__init__()

        self.first_conv = hk.Sequential([hk.Conv2D(output_channels=64,
                                                   kernel_shape=7, stride=1, padding="VALID"),
                                         hk.InstanceNorm(True, True),
                                         jax.nn.relu,
                                         ])

        self.layers = [
            self.first_conv,

            # Extend the number of channels incrementally.
            hk.Conv2D(output_channels=128, kernel_shape=3, stride=1),
            hk.InstanceNorm(True, True),
            jax.nn.relu,
            hk.Conv2D(output_channels=256, kernel_shape=3, stride=1),
            hk.InstanceNorm(True, True),
            jax.nn.relu,
        ]

        for _ in range(9):
            self.layers.append(ResnetBlock(256))

        # Final Transpose layer.
        self.layers.extend([
            hk.Conv2DTranspose(output_channels=3,
                               kernel_shape=7, stride=1, padding="VALID"),
            hk.InstanceNorm(True, True),
            jax.nn.tanh
        ])

        self.layers = hk.Sequential(self.layers)

    def __call__(self, x):
        """
        Call the model and run the input through the layers.

        @param x: The JAX array.
        @return: The outpput image in the range of -1 to 1.
        """
        x = self.layers(x)
        return x


@hk.without_apply_rng
@hk.transform
def generator(x):
    return Generator()(x)


if __name__ == "__main__":

    key = hk.PRNGSequence(0)

    random_input = jr.uniform(next(key), (1, 256, 256, 3))
    params = generator.init(next(key), random_input)

    sample_output = generator.apply(params, random_input)
    print("Output Shape:", sample_output.shape)
