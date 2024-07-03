import tensorflow as tf

class Layer:
    def __init__(self, keras_model):
        self.keras_model = keras_model

    def forward(self, input_data):
        # Assuming input_data is a NumPy array; convert it to a tf.Tensor
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        output_tensor = self.keras_model(input_tensor)
        return output_tensor.numpy()  # Convert back to NumPy array for compatibility

    def backward(self, input_data, output_error, learning_rate):
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        output_error_tensor = tf.convert_to_tensor(output_error, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            prediction = self.keras_model(input_tensor)
            loss = tf.reduce_mean(tf.square(prediction - output_error_tensor))

        # Calculate gradient of loss with respect to inputs
        input_gradient = tape.gradient(loss, input_tensor)

        # Calculate gradients of loss with respect to the model's trainable variables
        gradients = tape.gradient(loss, self.keras_model.trainable_variables)

        # Example of applying gradients to the model's trainable variables
        # Here, you should use an optimizer for practical applications
        for var, grad in zip(self.keras_model.trainable_variables, gradients):
            var.assign_sub(learning_rate * grad)

        return input_gradient.numpy()