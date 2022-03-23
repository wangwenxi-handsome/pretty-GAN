import tensorflow as tf
from tf.keras import layers


class GAN(tf.keras.Model):
    
    def __init__(
        self, 
        input_size = 256,
        latent_dim = 8 * 8 * 32,
    ):
        # 图片特征提取是不是有通用的backbone
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        self.discriminator = tf.keras.Sequential([
                tf.keras.Input(shape = (self.input_size, self.input_size, 3)),
                layers.Conv2D(32, 2, stride = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(64, 2, stride = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, 2, stride = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(64, 2, stride = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(32, 2, stride = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Reshape((self.latent_dim)),
            ],
            name = "discriminator",
        )
        
        self.generator = tf.keras.Sequential([
                tf.keras.Input(shape = (self.latent_dim)),
                layers.reshape((8, 8, 32)),
                layers.Conv2DTranspose(64, 2, stride = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(128, 2, stride = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(64, 2, stride = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(32, 2, stride = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(3, 2, stride = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
            ],
            name = "generator",
        )
        
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
    
    def build_loss(self, pred, label):
        return self.loss_fn(pred, label)

    def train_step(self, data):
        batch_size = data.shape[0]
        random_input = tf.random.normal(shape = (batch_size, self.latent_dim))
        gen_data = self.generator(random_input)
        gen_label = tf.ones((batch_size, 1))
        data = tf.concat([data, gen_data], axis = 0)
        label = tf.concat([tf.zeros((batch_size, 1)), gen_label], axis = 0)
        
        # train discriminator
        with tf.GradientTape as tape:
            dis_output = self.discriminator(data)
            dis_loss = self.build_loss(dis_output, label)
        dis_grads = tape.gradient(dis_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(dis_grads, self.discriminator.trainable_weights)
        )
        
        # train generator
        random_input = tf.random.normal(shape = (batch_size, self.latent_dim))
        gen_data = self.generator(random_input)
        gen_label = tf.zeros((batch_size * 2, 1))
        with tf.GradientTape as tape:
            gen_output = self.generator(gen_data)
            gen_loss = self.build_loss(gen_output, gen_label)
        gen_grads = tape.gradient(gen_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_weights)
        )
        
        return {"gen_loss": gen_loss, "dis_loss": dis_loss}