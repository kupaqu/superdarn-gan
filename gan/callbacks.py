import tensorflow as tf

class SaveCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, dst_dir: str):
        super().__init__()
        self.dst_dir = dst_dir

    def on_epoch_end(self, epoch, logs):
        epoch_dir = os.path.join(self.dst_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        self.model.g.save(os.path.join(epoch_dir, f'g.keras'))
        self.model.d.save(os.path.join(epoch_dir, f'd.keras'))
        self._plot_predictions(
            plot_path=os.path.join(epoch_dir, f'predictions.jpeg'),
            dataset=self.validation_data
        )
        self._plot_predictions(
            plot_path=os.path.join(epoch_dir, f'predictions.jpeg'),
            dataset=self.training_data
        )

    def on_train_end(self, logs):
        epoch_dir = os.path.join(self.dst_dir, 'train_result')
        os.makedirs(epoch_dir, exist_ok=True)
        self.model.g.save(os.path.join(epoch_dir, f'g.keras'))
        self.model.d.save(os.path.join(epoch_dir, f'd.keras'))
        self._plot_predictions(os.path.join(epoch_dir, f'predictions.jpeg'))

    
    def _plot_predictions(self, plot_path, dataset, n_examples=7):
        numpy_iterator = dataset.as_numpy_iterator()
        figure, axis = plt.subplots(2, n_examples)

        x, y = numpy_iterator.next()
        x = x[:, :, :, 0:1] * x[:, :, :, 1:2] # канал p_l*qflg

        ### предсказания
        fake_sample = self.model.g(x).numpy()
        fake_logits = self.model.d([x, fake_sample])
        real_logits = self.model.d([x, y])

        for i in range(n_examples):

            axis[0, i].imshow(y[i, :, :, 0])
            axis[0, i].set_title("real")

            axis[1, i].imshow(fake_sample[i, :, :, 0])
            axis[1, i].set_title("fake")

        plt.axis('off')
        plt.savefig(plot_path)
        plt.clf()
