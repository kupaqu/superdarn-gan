import tensorflow as tf

class GAN(tf.keras.Model):
    def __init__(self, d=None, g=None):
        super(GAN, self).__init__()

        if d is None:
            self.d = self.get_discriminator()
        else:
            self.d = d

        if g is None:
            self.g = self.get_generator()
        else:
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
    
    @staticmethod
    def bernoulli_mask(x, p=0.75):
        p = tf.constant([p])
        r = tf.random.uniform(shape=tf.shape(x), maxval=1)
        b = tf.math.greater(p, r)
        f = tf.cast(b, dtype=x.dtype)

        return tf.math.multiply(x, f)
    
    @staticmethod
    def mix(a, b):
        if not a.shape == b.shape:
            raise ValueError('Shapes of tensors are different')
        
        shape = tf.shape(a)
        target = tf.math.round(
            tf.random.uniform(
                shape=(shape[0], shape[2]),
                minval=0,
                maxval=1,
                dtype=a.dtype
            )
        )

        mask = tf.reshape(target, shape=(shape[0], 1, shape[2], 1))
        mask = tf.repeat(mask, repeats=shape[1], axis=1)
        mask = tf.repeat(mask, repeats=shape[3], axis=3)
        inv_mask = tf.math.substract(tf.ones_like(mask), mask)

        mixed = tf.math.add(
            tf.math.multiply(a, mask),
            tf.math.multiply(b, inv_mask)
        )

        return mixed, target

    def train_step(self, data):
        x, y = data
        x = x[:, :, :, 0:1] * x[:, :, :, 1:2] # канал p_l*qflg

        ### обучение дискриминатора
        x = self.bernoulli_mask(x) # шум на входе

        fake_sample = self.g(x)
        mixed_sample, mixed_target = self.mix(y, fake_sample)

        with tf.GradientTape() as tape:
            fake_logits = self.d([x, fake_sample], training=True)
            real_logits = self.d([x, y], training=True)
            mixed_logits = self.d([x, mixed_sample], training=True)

            real_loss = self.loss_fn(tf.ones_like(real_logits), real_logits) # 1 - настоящие значения
            fake_loss = self.loss_fn(tf.zeros_like(fake_logits), fake_logits) # 0 - фейковые значения
            mixed_loss = self.loss_fn(mixed_target, mixed_logits)

            d_loss = real_loss + fake_loss + mixed_loss

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
    # генератор
    def get_generator(self, regularizer_lambda=1e-5):
    
        # по каналу p_l
        p_l_history = tf.keras.layers.Input(shape=(100, 1080, 1))
    
        p_l = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(regularizer_lambda)
        )(p_l_history)
    
        p_l = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 12),
            dilation_rate=(1, 60),
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(regularizer_lambda)
        )(p_l)
    
        p_l = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            activation='relu',
            padding='same',
        )(p_l)
    
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 7),
            dilation_rate=(1, 60),
            activation='relu',
        )(p_l)
    
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            activation='relu',
            padding='same',
        )(x)
    
        output = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            activation='linear'
        )(x)
    
        generator = tf.keras.models.Model(p_l_history, output, name='generator')
    
        return generator
    
    # дискриминатор
    def get_discriminator(self, regularizer_lambda=1e-5):
    
        # по каналу p_l
        p_l_history = tf.keras.layers.Input(shape=(100, 1080, 1))
        p_l_target = tf.keras.layers.Input(shape=(100, 60, 1))
        p_l_concat = tf.keras.layers.Concatenate(axis=2)([p_l_history, p_l_target])
    
        p_l = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(regularizer_lambda)
        )(p_l_concat)
    
        p_l = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 13),
            dilation_rate=(1, 60),
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(regularizer_lambda)
        )(p_l)
    
        p_l = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            activation='relu',
            padding='same',
        )(p_l)
    
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 7),
            dilation_rate=(1, 60),
            activation='relu',
        )(p_l)#(x)
    
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            activation='relu',
            padding='same',
        )(x)
    
        x = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(100, 1),
            activation='sigmoid'
        )(x)
    
        output = tf.keras.layers.Flatten()(x)
    
        discriminator = tf.keras.models.Model([p_l_history, p_l_target], output, name='discriminator')
    
        return discriminator
