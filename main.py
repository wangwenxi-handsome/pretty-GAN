import tensorflow as tf 
from data import build_dataset
from model import GAN


data_path = "dataset/xinggan_face"
batch_size = 16
lr = 0.0001
epoch = 5


if __name__ == "__main__":
    # init
    dataset = build_dataset(data_path, batch_size)
    model = GAN()
    
    # compile
    model.compile(
        d_optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
        g_optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )
    
    # fit
    model.fit(dataset, epoch = 5)
