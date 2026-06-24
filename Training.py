from Preprocessing import tData, vData, fData, POINT_PAD_TOKEN, TEXT_PAD_TOKEN, VOCABSIZE, MAX_POINT_SEQ_LEN, MAX_TEXT_SEQ_LEN
import tensorflow as tf

WINDOW_NUM = 10
HIDDEN_SIZE = 400
PREDS_NUM = 20

class HandwritingSynthesisModel(tf.keras.Model):
    """
    Model for handwriting sythesis
    """

    def __init__(self):
        super().__init__()
        self.pointsMask = tf.keras.layers.Masking(mask_value=POINT_PAD_TOKEN)
        self.textMask = tf.keras.layers.Masking(mask_value=tf.one_hot(TEXT_PAD_TOKEN, VOCABSIZE))
        self.lstm0 = tf.keras.layers.LSTMCell(HIDDEN_SIZE)
        self.lstm1 = tf.keras.layers.LSTMCell(HIDDEN_SIZE)
        self.lstm2 = tf.keras.layers.LSTMCell(HIDDEN_SIZE)
        self.windowDense = tf.keras.layers.Dense(3 * WINDOW_NUM)
        self.mdn = tf.keras.layers.Dense(6 * PREDS_NUM + 1)

 #   @tf.function
    def call(self, inputs, training=False):
        pointsInput, textInput = inputs
        batchSize = tf.shape(pointsInput)[0]

        textInput = tf.one_hot(textInput, VOCABSIZE)
        pointsInput = self.pointsMask(pointsInput)
        textInput = self.textMask(textInput)
        
        # print("Points input", tf.shape(pointsInput))
        # print("Text input", tf.shape(textInput))

        w0 = tf.zeros((batchSize, VOCABSIZE))
        states00 = [tf.zeros((batchSize, HIDDEN_SIZE)), tf.zeros((batchSize, HIDDEN_SIZE))]
        states01 = [tf.zeros((batchSize, HIDDEN_SIZE)), tf.zeros((batchSize, HIDDEN_SIZE))]
        states02 = [tf.zeros((batchSize, HIDDEN_SIZE)), tf.zeros((batchSize, HIDDEN_SIZE))]
        kappa = tf.zeros((batchSize, WINDOW_NUM))
        u = tf.reshape(tf.range(MAX_TEXT_SEQ_LEN, dtype=tf.float32), (1,1,-1))
        outputs = []

        # print("w0", tf.shape(w0))
        # print("states00[0]", tf.shape(states00[0]))

        for timestep in range(MAX_POINT_SEQ_LEN):
         #   print(timestep)
            point = pointsInput[:, timestep, :] #shape(batch, 3)
         #   expandedWindow = tf.expand_dims(w0, 1) #shape(batch, 1, VOCABSIZE)
            pointWindow = tf.concat([point, w0], 1) #shape(batch,VOCABSIZE+3)
            output, states = self.lstm0(pointWindow, states00) #[hidden,cell]
            states00 = states
            alphaHat, betaHat, kappaHat = tf.split(self.windowDense(states[0]), 3, axis=1)
            kappa = kappa + tf.exp(kappaHat)
            alpha = tf.exp(alphaHat)
            beta = tf.exp(betaHat)
            phi = tf.reshape(tf.reduce_sum(tf.exp(-tf.reshape(beta, (batchSize,10,1)) * (tf.reshape(kappa, (batchSize,10,1)) - u) ** 2) * tf.reshape(alpha, (batchSize,10,1)), axis=1), (batchSize,-1,1))
            w0 = tf.reduce_sum(phi * textInput, axis=1)
            output, states = self.lstm1(tf.concat([point, w0, states00[0]], axis=1), states01)
            states01 = states
            output, states = self.lstm2(tf.concat([point, w0, states01[0]], axis=1), states02)
            states02 = states
            outputs.append(states02[0])
        
        final = tf.stack(outputs, axis=1)
        final = self.mdn(final)
        pi, mux, muy, sigmax, sigmay, rho, penup = tf.split(final, [20,20,20,20,20,20,1], axis=2)
        
        return tf.nn.softmax(pi), mux, muy, tf.exp(sigmax), tf.exp(sigmay), tf.nn.tanh(rho), tf.nn.sigmoid(penup)

if __name__ == "__main__":
    for point, text in tData.take(1):
        model = HandwritingSynthesisModel()
        model((point, text))