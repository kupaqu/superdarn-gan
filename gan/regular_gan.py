import tensorflow as tf

class GAN(tf.keras.Model):
    def __init__(self, d, g):
        super(GAN, self).__init__()

        self.d = d
        self.g = g

        self.d_loss_tracker = tf.keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = tf.keras.metrics.Mean(name='g_loss')
        self.g_mae_tracker = tf.keras.metrics.Mean(name='g_mae')

    @property
    def metrics(self):
        return [
            self.d_loss_tracker,
            self.g_loss_tracker,
            self.g_mae_tracker,
        ]

    def compile(self, d_opt, g_opt, loss_fn):
        super(GAN, self).compile()
        self.d_opt = d_opt
        self.g_opt = g_opt
        self.loss_fn = loss_fn
        self.g_mae_fn = tf.keras.losses.MeanAbsoluteError()

    def train_step(self, data):
        x, y = data
        x = x[:, :, :, 0:1] * x[:, :, :, 1:2] # канал p_l*qflg

        ### обучение дискриминатора
        fake_sample = self.g(x)
        with tf.GradientTape() as tape:
            fake_logits = self.d([x, fake_sample], training=True)
            real_logits = self.d([x, y], training=True)

            real_loss = self.loss_fn(tf.ones_like(real_logits), real_logits) # 1 - настоящие значения
            fake_loss = self.loss_fn(tf.zeros_like(fake_logits), fake_logits) # 0 - фейковые значения

            d_loss = real_loss + fake_loss

        grads = tape.gradient(d_loss, self.d.trainable_weights)
        self.d_opt.apply_gradients(
            zip(grads, self.d.trainable_weights)
        )

        ### обучение генератора
        with tf.GradientTape() as tape:
            fake_sample = self.g(x, training=True)
            fake_logits = self.d([x, fake_sample])

            g_loss = self.loss_fn(tf.ones_like(fake_logits), fake_logits) # фейковые значения должны быть приняты дискриминатором за настоящие
            g_mae = self.g_mae_fn(y, fake_sample)

        grads = tape.gradient(g_loss, self.g.trainable_weights)
        self.g_opt.apply_gradients(zip(grads, self.g.trainable_weights))

        ### обновляем трекеры
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.g_mae_tracker.update_state(g_mae)

        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
            "g_mae": self.g_mae_tracker.result()
        }

    def test_step(self, data):
        x, y = data
        x = x[:, :, :, 0:1] * x[:, :, :, 1:2] # канал p_l*qflg

        ### предсказания
        fake_sample = self.g(x)
        fake_logits = self.d([x, fake_sample])
        real_logits = self.d([x, y])

        ### метрики дискриминатора
        real_loss = self.loss_fn(tf.ones_like(real_logits), real_logits) # 1 - настоящие значения
        fake_loss = self.loss_fn(tf.zeros_like(fake_logits), fake_logits) # 0 - фейковые значения
        d_loss = real_loss + fake_loss

        ### метрики генератора
        g_loss = self.loss_fn(tf.ones_like(fake_logits), fake_logits) # фейковые значения должны быть приняты дискриминатором за настоящие
        g_mae = self.g_mae_fn(y, fake_sample)

        ### обновляем трекеры
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.g_mae_tracker.update_state(g_mae)

        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
            "g_mae": self.g_mae_tracker.result()
        }
