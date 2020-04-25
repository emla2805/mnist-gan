from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.layers import (
    Conv2DTranspose,
    LeakyReLU,
    BatchNormalization,
    Dense,
    Conv2D,
    Reshape,
    Dropout,
    Flatten,
)

from net import GAN

AUTOTUNE = tf.data.experimental.AUTOTUNE
NUM_EXAMPLES_TO_GENERATE = 16

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--generator-lr", default=1e-4, type=float)
    parser.add_argument("--discriminator-lr", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--latent-dim", default=100, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    args = parser.parse_args()

    def convert_types(image, _):
        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5  # Normalize the images to [-1, 1]
        return image

    dataset = tfds.load("mnist", as_supervised=True, split=tfds.Split.TRAIN)
    dataset = (
        dataset.map(convert_types)
        .cache()
        .shuffle(1000)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )

    generator = tf.keras.models.Sequential(
        [
            Dense(7 * 7 * 256, use_bias=False, input_shape=(args.latent_dim,)),
            BatchNormalization(),
            LeakyReLU(),
            Reshape((7, 7, 256)),
            Conv2DTranspose(
                128, (5, 5), strides=(1, 1), padding="same", use_bias=False
            ),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(
                64, (5, 5), strides=(2, 2), padding="same", use_bias=False
            ),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(
                1,
                (5, 5),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                activation="tanh",
            ),
        ]
    )

    discriminator = tf.keras.models.Sequential(
        [
            Conv2D(
                64,
                (5, 5),
                strides=(2, 2),
                padding="same",
                input_shape=[28, 28, 1],
            ),
            LeakyReLU(),
            Dropout(0.3),
            Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
            LeakyReLU(),
            Dropout(0.3),
            Flatten(),
            Dense(1),
        ]
    )

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(args.generator_lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(args.discriminator_lr)

    gan = GAN(discriminator, generator, 100)
    gan.compile(discriminator_optimizer, generator_optimizer, cross_entropy)

    # Seed for generating images
    seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, args.latent_dim])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=args.log_dir, profile_batch=0
    )
    file_writer = tf.summary.create_file_writer(args.log_dir)

    def log_images(epoch, logs):
        gen_images = gan.generator(seed)
        images = tf.reshape(gen_images * 0.5 + 0.5, (-1, 28, 28, 1))

        with file_writer.as_default():
            tf.summary.image(
                "Generated digits",
                images,
                max_outputs=NUM_EXAMPLES_TO_GENERATE,
                step=epoch,
            )

    image_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_images)

    gan.fit(
        dataset,
        epochs=args.epochs,
        callbacks=[tensorboard_callback, image_callback],
    )
