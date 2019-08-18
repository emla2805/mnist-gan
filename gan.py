import time
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

AUTOTUNE = tf.data.experimental.AUTOTUNE
NUM_EXAMPLES_TO_GENERATE = 16

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where model summaries and checkpoints are stored",
    )
    parser.add_argument(
        "--generator-lr",
        default=1e-4,
        type=float,
        help="Generator learning rate",
    )
    parser.add_argument(
        "--discriminator-lr",
        default=1e-4,
        type=float,
        help="Discriminator learning rate",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, help="Batch Size when training"
    )
    parser.add_argument(
        "--noise-dim",
        default=100,
        type=int,
        help="Size of the noise dimension",
    )
    parser.add_argument(
        "--epochs", default=20, type=int, help="Number of training epochs"
    )
    args = parser.parse_args()

    def convert_types(image, _):
        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5  # Normalize the images to [-1, 1]
        return image

    dataset = tfds.load("mnist", as_supervised=True, split=tfds.Split.TRAIN)
    dataset = (
        dataset.map(convert_types)
        .cache()
        .shuffle(60000)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )

    generator = tf.keras.models.Sequential(
        [
            Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
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

    # Loss helper function
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    generator_optimizer = tf.keras.optimizers.Adam(args.generator_lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(args.discriminator_lr)

    ckpt = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    manager = tf.train.CheckpointManager(ckpt, args.log_dir, max_to_keep=1)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    gen_loss = tf.keras.metrics.Mean(name="gen_loss")
    disc_loss = tf.keras.metrics.Mean(name="disc_loss")

    summary_writer = tf.summary.create_file_writer(args.log_dir)

    # Seed for generating images
    seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, args.noise_dim])

    @tf.function
    def train_step(images):
        noise = tf.random.normal([args.batch_size, args.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            g_loss = generator_loss(fake_output)
            d_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            g_loss, generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            d_loss, discriminator.trainable_variables
        )

        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        )

        gen_loss(g_loss)
        disc_loss(d_loss)

    for epoch in range(args.epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        with summary_writer.as_default():
            tf.summary.scalar("loss/generator", gen_loss.result(), step=epoch)
            tf.summary.scalar(
                "loss/discriminator", disc_loss.result(), step=epoch
            )

        if (epoch + 1) % 5 == 0:
            save_path = manager.save()
            print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

        # Save images
        gen_images = generator(seed, training=False)
        images = tf.reshape(gen_images * 0.5 + 0.5, (-1, 28, 28, 1))
        with summary_writer.as_default():
            tf.summary.image(
                "Generated digits",
                images,
                max_outputs=NUM_EXAMPLES_TO_GENERATE,
                step=epoch,
            )

        template = "Epoch {} took {:.0f} sec, Gen Loss: {}, Disc Loss: {}"
        print(
            template.format(
                epoch + 1,
                time.time() - start,
                gen_loss.result(),
                disc_loss.result(),
            )
        )

        # Reset metrics every epoch
        gen_loss.reset_states()
        disc_loss.reset_states()
