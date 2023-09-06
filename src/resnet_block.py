import jax
import jax.numpy as jnp
import haiku as hk


class ResnetBlock(hk.Module):

    def __init__(self, channels):
        super(ResnetBlock, self).__init__()

        self.conv1 = hk.Sequential([
            hk.Conv2D(output_channels=channels, kernel_shape=3, stride=1, padding="VALID"),
            hk.InstanceNorm(True, True),
            jax.nn.relu
        ])

        self.conv2 = hk.Sequential([
            hk.Conv2DTranspose(output_channels=channels, kernel_shape=3, stride=1, padding="VALID"),
            hk.InstanceNorm(True, True),
            jax.nn.relu
        ])

    def __call__(self, x):
        x_ = self.conv1(x)
        x_ = self.conv2(x_)
        return x + x_
