import tensorflow as tf
from tensorflow.keras import layers


class BilinearInterpolation(layers.Layer):
    def __init__(self, output_size, **kwargs):
        super(BilinearInterpolation, self).__init__(**kwargs)
        self.output_size = output_size

    def get_config(self):
        return {
            'output_size': self.output_size,
        }

    def call(self, inputs):
        x, transformation = inputs
        output = self._transform(x, transformation, self.output_size)
        return output

    def _make_regular_grids(self, batch_size, height, width):
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinate = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, [-1])
        y_coordinate = tf.reshape(y_coordinate, [-1])
        ones = tf.ones_like(x_coordinates)
        grid = tf.concat([x_coordinates, y_coordinate, ones], 0)

        grid = tf.reshape(grid, [-1])

        grids = tf.tile(grid, tf.stack([batch_size]))
        return tf.reshape(grids, (batch_size, 3, height * width))

    def _interpolate(self, image, sampled_grids, output_size):
        batch_size, height, width, num_channels = image.shape
        batch_size = tf.shape(image)[0]
        x = tf.cast(tf.reshape(sampled_grids[:, 0:1, :], [-1]), dtype=tf.float32)
        y = tf.cast(tf.reshape(sampled_grids[:, 1:2, :], [-1]), dtype=tf.float32)

        x = .5 * (x + 1.0) * tf.cast(width, dtype=tf.float32)
        y = .5 * (y + 1.0) * tf.cast(height, dtype=tf.float32)

        x0 = tf.cast(x, tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(y, tf.int32)
        y1 = y0 + 1

        max_x = int(tf.keras.backend.int_shape(image)[2] - 1)
        max_y = int(tf.keras.backend.int_shape(image)[1] - 1)

        x0 = tf.keras.backend.clip(x0, 0, max_x)
        x1 = tf.keras.backend.clip(x1, 0, max_x)
        y0 = tf.keras.backend.clip(y0, 0, max_y)
        y1 = tf.keras.backend.clip(y1, 0, max_y)

        pixels_batch = tf.keras.backend.arange(0, batch_size) * (height * width)
        pixels_batch = tf.keras.backend.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        base = tf.keras.backend.repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = tf.keras.backend.flatten(base)

        # base_y0 = base + (y0 * width)
        base_y0 = y0 * width
        base_y0 = base + base_y0
        # base_y1 = base + (y1 * width)
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.keras.backend.reshape(image, shape=(-1, num_channels))
        flat_image = tf.keras.backend.cast(flat_image, dtype='float32')
        pixel_values_a = tf.keras.backend.gather(flat_image, indices_a)
        pixel_values_b = tf.keras.backend.gather(flat_image, indices_b)
        pixel_values_c = tf.keras.backend.gather(flat_image, indices_c)
        pixel_values_d = tf.keras.backend.gather(flat_image, indices_d)

        x0 = tf.keras.backend.cast(x0, 'float32')
        x1 = tf.keras.backend.cast(x1, 'float32')
        y0 = tf.keras.backend.cast(y0, 'float32')
        y1 = tf.keras.backend.cast(y1, 'float32')

        area_a = tf.keras.backend.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.keras.backend.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.keras.backend.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.keras.backend.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    def _transform(self, X, affine_transformation, output_size):
        batch_size, num_channels = tf.shape(X)[0], X.shape[3]
        transformations = tf.reshape(affine_transformation, (batch_size, 2, 3))
        regular_grids = self._make_regular_grids(batch_size, *output_size)
        sampled_grids = tf.keras.backend.batch_dot(transformations, regular_grids)

        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        new_shape = (batch_size, output_size[0], output_size[1], num_channels)
        interpolated_image = tf.keras.backend.reshape(interpolated_image, new_shape)
        return interpolated_image
