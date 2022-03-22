from multiprocessing.sharedctypes import Value
import os
import tensorflow as tf
from tqdm import trange


def precessor(file, format="jpeg"):
    pic = tf.io.read_file(file)
    if format == "jpeg":
        pic = tf.image.decode_jpeg(pic)
    else:
        raise ValueError("format is not supported")
    return pic


def build_dataset(
    path, 
    batch_size = 32, 
):
    # get file name
    file = os.listdir(path)
    file = [os.path.join(path, i) for i in file]
    
    dataset = tf.data.Dataset.from_tensor_slices(file)
    dataset = dataset.map(precessor, num_parallel_calls=2).batch(batch_size).prefetch(1)
    return dataset
        

if __name__ == "__main__":
    path = "dataset/xinggan_face"
    train_data = build_dataset(path)
    for i in train_data:
        print(i)