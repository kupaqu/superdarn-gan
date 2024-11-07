import tensorflow as tf

class GradientVariance(tf.keras.losses.Loss):
    """Class for calculating GV loss between two images.
    Original: https://github.com/lusinlu/gradient-variance-loss.
    :parameter
    patch_size : int, size of the patches extracted from the gt and predicted images
    reduction : str, how to reduce the loss (AUTO, NONE, SUM, SUM_OVER_BATCH_SIZE)
    """
    def __init__(self, patch_size=8, reduction=tf.keras.losses.Reduction.AUTO):
        super(GradientVariance, self).__init__(reduction=reduction)
        self.patch_size = patch_size
        # Sobel kernel for the gradient map calculation
        self.kernel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        self.kernel_y = tf.constant([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]], dtype=tf.float32)
        self.kernel_x = tf.expand_dims(tf.expand_dims(self.kernel_x, -1), -1)
        self.kernel_y = tf.expand_dims(tf.expand_dims(self.kernel_y, -1), -1)

    def call(self, y_true, y_pred):
        # Initialize loss
        total_loss = 0.0
        
        num_channels = y_true.shape[-1]  # Get the number of channels

        for c in range(num_channels):
            gray_output = y_pred[..., c:c+1]  # Process channel c
            gray_target = y_true[..., c:c+1]

            # Calculate the gradient maps of x and y directions
            gx_target = tf.nn.conv2d(gray_target, self.kernel_x, strides=[1, 1, 1, 1], padding='SAME')
            gy_target = tf.nn.conv2d(gray_target, self.kernel_y, strides=[1, 1, 1, 1], padding='SAME')
            gx_output = tf.nn.conv2d(gray_output, self.kernel_x, strides=[1, 1, 1, 1], padding='SAME')
            gy_output = tf.nn.conv2d(gray_output, self.kernel_y, strides=[1, 1, 1, 1], padding='SAME')

            # Extracting patches
            gx_target_patches = tf.image.extract_patches(images=gx_target, sizes=[1, self.patch_size, self.patch_size, 1],
                                                          strides=[1, self.patch_size, self.patch_size, 1],
                                                          rates=[1, 1, 1, 1], padding='VALID')
            gy_target_patches = tf.image.extract_patches(images=gy_target, sizes=[1, self.patch_size, self.patch_size, 1],
                                                          strides=[1, self.patch_size, self.patch_size, 1],
                                                          rates=[1, 1, 1, 1], padding='VALID')
            gx_output_patches = tf.image.extract_patches(images=gx_output, sizes=[1, self.patch_size, self.patch_size, 1],
                                                          strides=[1, self.patch_size, self.patch_size, 1],
                                                          rates=[1, 1, 1, 1], padding='VALID')
            gy_output_patches = tf.image.extract_patches(images=gy_output, sizes=[1, self.patch_size, self.patch_size, 1],
                                                          strides=[1, self.patch_size, self.patch_size, 1],
                                                          rates=[1, 1, 1, 1], padding='VALID')

            # Calculation of variance of each patch
            var_target_x = tf.math.reduce_variance(gx_target_patches, axis=-1)
            var_output_x = tf.math.reduce_variance(gx_output_patches, axis=-1)
            var_target_y = tf.math.reduce_variance(gy_target_patches, axis=-1)
            var_output_y = tf.math.reduce_variance(gy_output_patches, axis=-1)

            # Loss function as a MSE between variances of patches extracted from gradient maps
            gradvar_loss = tf.keras.losses.mean_squared_error(var_target_x, var_output_x) + \
                           tf.keras.losses.mean_squared_error(var_target_y, var_output_y)

            total_loss += gradvar_loss  # Accumulate loss for all channels

        return total_loss