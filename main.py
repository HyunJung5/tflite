import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('./mnist_cnn/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
open("saved_model/my_model.quant.tflite", "wb").write(tflite_quant_model)
