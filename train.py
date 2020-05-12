from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Flatten,
    LeakyReLU,
    ReLU,
)
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import TensorBoard

from net import GAN

AUTOTUNE = tf.data.experimental.AUTOTUNE
NUM_EXAMPLES_TO_GENERATE = 16

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--generator-lr", default=2e-4, type=float)
    parser.add_argument("--discriminator-lr", default=2e-4, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--latent-dim", default=100, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    args = parser.parse_args()

    def normalize(img, _):
        img = tf.image.resize(img, size=(32, 32))
        img = (img - 127.5) / 127.5  # Normalize the images to [-1, 1]
        return img

    dataset = (
        tfds.load("mnist", as_supervised=True, split="train+test")
        .map(normalize)
        .cache()
        .shuffle(1024)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )

    init = RandomNormal(stddev=0.02)
    generator = tf.keras.models.Sequential(
        [
            Conv2DTranspose(
                128 * 4,
                (4, 4),
                strides=(1, 1),
                use_bias=False,
                kernel_initializer=init,
                input_shape=(1, 1, args.latent_dim),
            ),
            BatchNormalization(),
            ReLU(),  # (None, 4, 4, 128 * 4)
            Conv2DTranspose(
                128 * 2,
                (4, 4),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                kernel_initializer=init,
            ),
            BatchNormalization(),
            ReLU(),  # (None, 8, 8, 128 * 2)
            Conv2DTranspose(
                128,
                (4, 4),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                kernel_initializer=init,
            ),
            BatchNormalization(),
            ReLU(),  # (None, 16, 16, 128)
            Conv2DTranspose(
                1,
                (4, 4),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                kernel_initializer=init,
                activation="tanh",
            ),  # (None, 32, 32, 1)
        ]
    )

    discriminator = tf.keras.models.Sequential(
        [
            Conv2D(
                128,
                (4, 4),
                strides=(2, 2),
                padding="same",
                input_shape=(32, 32, 1),
                use_bias=False,
                kernel_initializer=init,
            ),
            LeakyReLU(0.2),  # (None, 16, 16, 128)
            Conv2D(
                128 * 2,
                (4, 4),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                kernel_initializer=init,
            ),
            BatchNormalization(),
            LeakyReLU(0.2),  # (None, 8, 8, 128 * 2)
            Conv2D(
                128 * 4,
                (4, 4),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                kernel_initializer=init,
            ),
            BatchNormalization(),
            LeakyReLU(0.2),  # (None, 4, 4, 128 * 4)
            Conv2D(
                1,
                (4, 4),
                strides=(1, 1),
                use_bias=False,
                kernel_initializer=init,
            ),
            Flatten(),
        ]
    )

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(
        args.generator_lr, beta_1=0.5
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        args.discriminator_lr, beta_1=0.5
    )

    gan = GAN(discriminator, generator, args.latent_dim)
    gan.compile(discriminator_optimizer, generator_optimizer, cross_entropy)

    # Seed for generating images
    seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, 1, 1, args.latent_dim])

    class GANMonitor(tf.keras.callbacks.Callback):
        def __init__(self, log_dir, latent_vectors):
            super().__init__()
            self.file_writer = tf.summary.create_file_writer(log_dir)
            self.latent_vectors = latent_vectors

        def on_epoch_end(self, epoch, logs=None):
            generated_images = self.model.generator(self.latent_vectors)

            with self.file_writer.as_default():
                tf.summary.image(
                    "Generated Images",
                    generated_images,
                    max_outputs=NUM_EXAMPLES_TO_GENERATE,
                    step=epoch,
                )

    gan.fit(
        dataset,
        epochs=args.epochs,
        callbacks=[
            TensorBoard(log_dir=args.log_dir, profile_batch=0),
            GANMonitor(log_dir=args.log_dir, latent_vectors=seed),
        ],
    )
