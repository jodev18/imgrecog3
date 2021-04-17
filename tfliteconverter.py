import tensorflow as tf

model = tf.keras.models.load_model('best_model_dataflair3.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("tflitemodel_posedetect.tflite", "wb").write(tflite_model)