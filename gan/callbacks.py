import tensorflow as tf
import matplotlib.pyplot as plt
import os

class SaveCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, dst_dir: str, training_data: tf.data.Dataset, validation_data: tf.data.Dataset):
        super().__init__()
        self.dst_dir = dst_dir
        self.training_data = training_data
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs):
        epoch_dir = os.path.join(self.dst_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        self.model.g.save(os.path.join(epoch_dir, f'g.keras'))
        self.model.d.save(os.path.join(epoch_dir, f'd.keras'))
        self._plot_predictions(
            plot_path=os.path.join(epoch_dir, f'train_predictions.jpeg'),
            dataset=self.training_data
        )
        self._plot_predictions(
            plot_path=os.path.join(epoch_dir, f'val_predictions.jpeg'),
            dataset=self.validation_data
        )

    def on_train_end(self, logs):
        self.on_epoch_end('end')

    def _plot_predictions(self, plot_path, dataset, n_examples=7):
        dataset.repeat()
        numpy_iterator = dataset.as_numpy_iterator()
        x, y = numpy_iterator.next()
        x = x[:, :, :, 0:1] * x[:, :, :, 1:2] # канал p_l*qflg
        dataset.repeat()

        ### предсказания
        fake_sample = self.model.g(x).numpy()
        fake_logits = self.model.d([x, fake_sample])
        real_logits = self.model.d([x, y])

        figure, axis = plt.subplots(2, n_examples)
        for i in range(n_examples):

            axis[0, i].imshow(y[i, :, :, 0])
            axis[0, i].set_title("real")

            axis[1, i].imshow(fake_sample[i, :, :, 0])
            axis[1, i].set_title("fake")

        plt.axis('off')
        plt.savefig(plot_path)
        plt.clf()
