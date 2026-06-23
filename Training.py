from Preprocessing import tData, vData, fData, POINT_PAD_TOKEN, TEXT_PAD_TOKEN, VOCABSIZE
import tensorflow as tf

class HandwritingSynthesisModel(tf.keras.Model):
    """
    Model for handwriting sythesis
    """

    def __init__(self):
        super().__init__()
        self.pointsMask = tf.keras.layers.Masking(mask_value=POINT_PAD_TOKEN)
        self.textMask = tf.keras.layers.Masking(mask_value=TEXT_PAD_TOKEN)

    def call(self, inputs, training=False):
        pointsInput, textInput = inputs
        pointsInput = self.pointsMask(pointsInput)
        textInput = self.textMask(textInput)
        textInput = tf.one_hot(textInput, VOCABSIZE)