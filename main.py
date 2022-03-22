import tensorflow as tf 
from data import build_dataset
from model import GAN


data_path = "dataset/xinggan_face"
batch_size = 32
lr = 0.01


if __name__ == "__main__":
    # init
    dataset = build_dataset(data_path, batch_size)
    model = GAN()
    
    # compile
    opt = tf.keras.optimizers.Adam(lr = lr, )
    model.compile(
        optimizer = opt,
    )
    
    # fit
    model.fit(dataset, )
    
    