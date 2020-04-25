import tensorflow as tf


class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, discriminator_optimizer, generator_optimizer, loss_fn):
        super(GAN, self).compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(random_latent_vectors)

            real_predictions = self.discriminator(real_images)
            fake_predictions = self.discriminator(generated_images)

            g_loss = self._generator_loss(fake_predictions)
            d_loss = self._discriminator_loss(
                real_predictions, fake_predictions
            )

        gradients_of_generator = gen_tape.gradient(
            g_loss, self.generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(
                gradients_of_discriminator,
                self.discriminator.trainable_variables,
            )
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

    def _discriminator_loss(self, real_predictions, fake_predictions):
        real_labels = tf.ones_like(real_predictions)
        fake_labels = tf.zeros_like(fake_predictions)

        # Add random noise to the labels - important trick!
        real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))
        fake_labels += 0.05 * tf.random.uniform(tf.shape(fake_labels))

        real_loss = self.loss_fn(real_labels, real_predictions)
        fake_loss = self.loss_fn(fake_labels, fake_predictions)
        total_loss = real_loss + fake_loss
        return total_loss

    def _generator_loss(self, fake_predictions):
        fake_labels = tf.ones_like(fake_predictions)
        fake_labels += 0.05 * tf.random.uniform(tf.shape(fake_labels))
        return self.loss_fn(fake_labels, fake_predictions)
