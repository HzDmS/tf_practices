import tensorflow as tf
import tensorflow_datasets as tf_ds

from layers.dense_layer import DenseLayer
from models.model import Model

tf.random.set_seed(42)


def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.
    image = tf.reshape(image, [image.shape[0] * image.shape[1] * image.shape[2]])
    return image, label


if __name__ == '__main__':
    BATCH_SIZE = 64
    (ds_train, ds_test), ds_info = tf_ds.load(
        "mnist", split=["train", "test"], shuffle_files=True, as_supervised=True, with_info=True)
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    model = Model([DenseLayer(name="dense0", shape=[28 * 28, 32]), DenseLayer(name="dense1", shape=[32, 10])])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for ep in range(36):

        metrics_train = tf.keras.metrics.SparseCategoricalAccuracy()

        for x_batch, y_batch in ds_train:
            with tf.GradientTape() as tape:
                logits = model(x_batch)
                y_pred = tf.nn.softmax(logits)
                metrics_train.update_state(y_batch, y_pred)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_batch, logits)
            grad = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grad, model.variables))

        metrics_test = tf.keras.metrics.SparseCategoricalAccuracy()

        for x_test_batch, y_test_batch in ds_test:
            logits = model(x_test_batch)
            test_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_test_batch, logits)
            y_pred = tf.nn.softmax(logits)
            metrics_test.update_state(y_test_batch, y_pred)

        print(f"Epoch: {ep},"
              f" Training Loss: {tf.reduce_mean(loss):.3f},"
              f" Training Acc: {metrics_train.result():.3f},"
              f" Testing Loss: {tf.reduce_mean(test_loss):.3f},"
              f" Testing Acc: {metrics_test.result():.3f}")
