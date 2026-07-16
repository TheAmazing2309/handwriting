from Preprocessing import (tData, vData, fData, datasetNorms, 
                           POINT_PAD_TOKEN, TEXT_PAD_TOKEN, VOCABSIZE, MAX_POINT_SEQ_LEN, MAX_TEXT_SEQ_LEN, CHECKPOINT_PATH,
                        BATCH_SIZE, GRAPH_PATH, visualizeSample, samplePoint, visualizeStrokes)
from Loss import loss
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

WINDOW_NUM = 10
HIDDEN_SIZE = 400
PREDS_NUM = 20
NUM_BATCHES = sum(1 for _ in tData)

EPOCHS = 20


@tf.custom_gradient
def clip_gradient(x, clip_value):
    """
    Identity on the forward pass; clips the incoming gradient element-wise to
    [-clip_value, clip_value] on the backward pass (Graves 2013, Sec. 2.1/4.2).
    """
    def grad(dy):
        return tf.clip_by_value(dy, -clip_value, clip_value), None
    return x, grad


class PeepholeLSTMCell(tf.keras.layers.Layer):
    """
    LSTM cell with peephole connections from the cell state to the gates,
    matching Eqs. 7-11 of Graves 2013.
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = [units, units]
        self.output_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, 4 * self.units), name="W", initializer="glorot_uniform")
        self.U = self.add_weight(shape=(self.units, 4 * self.units), name="U", initializer="orthogonal")
        self.peephole = self.add_weight(shape=(3 * self.units,), name="peephole", initializer="zeros")
        self.bias = self.add_weight(shape=(4 * self.units,), name="bias", initializer="zeros")
        self.built = True

    def call(self, inputs, states):
        h_tm1, c_tm1 = states
        z = tf.matmul(inputs, self.W) + tf.matmul(h_tm1, self.U) + self.bias
        zi, zf, zc, zo = tf.split(z, 4, axis=1)
        peepI, peepF, peepO = tf.split(self.peephole, 3)

        i = tf.sigmoid(zi + peepI * c_tm1)
        f = tf.sigmoid(zf + peepF * c_tm1)
        c = f * c_tm1 + i * tf.tanh(zc)
        o = tf.sigmoid(zo + peepO * c)
        h = o * tf.tanh(c)
        return h, [h, c]


class HandwritingSynthesisModel(tf.keras.Model):
    """
    Model for handwriting sythesis
    """

    def __init__(self):
        super().__init__()
        self.pointsMask = tf.keras.layers.Masking(mask_value=POINT_PAD_TOKEN)
        self.textMask = tf.keras.layers.Masking(mask_value=tf.one_hot(TEXT_PAD_TOKEN, VOCABSIZE))
        self.lstm0 = PeepholeLSTMCell(HIDDEN_SIZE)
        self.lstm1 = PeepholeLSTMCell(HIDDEN_SIZE)
        self.lstm2 = PeepholeLSTMCell(HIDDEN_SIZE)
        self.windowDense = tf.keras.layers.Dense(3 * WINDOW_NUM)
        self.mdn = tf.keras.layers.Dense(6 * PREDS_NUM + 1)

    def call(self, inputs, training=False):
        pointsInput, textInput = inputs
        batchSize = tf.shape(pointsInput)[0]

        # true (non-[PAD]) character positions, so the window never attends to padding
        textValidMask = tf.cast(tf.not_equal(textInput, TEXT_PAD_TOKEN), tf.float32)
        textInput = tf.one_hot(textInput, VOCABSIZE)
        pointsInput = self.pointsMask(pointsInput)
        textInput = self.textMask(textInput)
        
        # print("Points input", tf.shape(pointsInput))
        # print("Text input", tf.shape(textInput))
        # tf.print("Points mask:", pointsInput._keras_mask[0])
        # tf.print("Text mask:", textInput._keras_mask[0])

        w0 = tf.zeros((batchSize, VOCABSIZE))
        states00 = [tf.zeros((batchSize, HIDDEN_SIZE)), tf.zeros((batchSize, HIDDEN_SIZE))]
        states01 = [tf.zeros((batchSize, HIDDEN_SIZE)), tf.zeros((batchSize, HIDDEN_SIZE))]
        states02 = [tf.zeros((batchSize, HIDDEN_SIZE)), tf.zeros((batchSize, HIDDEN_SIZE))]
        kappa = tf.zeros((batchSize, WINDOW_NUM))
        u = tf.reshape(tf.range(MAX_TEXT_SEQ_LEN, dtype=tf.float32), (1,1,-1))
        outputs = []

        # print("w0", tf.shape(w0))
        # print("states00[0]", tf.shape(states00[0]))

        seqLen = pointsInput.shape[1]
        for timestep in range(seqLen):
            point = pointsInput[:, timestep, :] #shape(batch, 3)
         #   expandedWindow = tf.expand_dims(w0, 1) #shape(batch, 1, VOCABSIZE)
            pointWindow = clip_gradient(tf.concat([point, w0], 1), 10.0) #shape(batch,VOCABSIZE+3)
            output, states = self.lstm0(pointWindow, states00) #[hidden,cell]
            states00 = states
            alphaHat, betaHat, kappaHat = tf.split(self.windowDense(states[0]), 3, axis=1)
            kappa = kappa + tf.exp(kappaHat)
            alpha = tf.exp(alphaHat)
            beta = tf.exp(betaHat)
            phi = tf.reshape(tf.reduce_sum(tf.exp(-tf.reshape(beta, (batchSize,10,1)) * (tf.reshape(kappa, (batchSize,10,1)) - u) ** 2) * tf.reshape(alpha, (batchSize,10,1)), axis=1), (batchSize,-1,1))
            phi = phi * tf.expand_dims(textValidMask, -1) # never attend to [PAD] characters
            w0 = tf.reduce_sum(phi * textInput, axis=1)
            lstm1Input = clip_gradient(tf.concat([point, w0, states00[0]], axis=1), 10.0)
            output, states = self.lstm1(lstm1Input, states01)
            states01 = states
            lstm2Input = clip_gradient(tf.concat([point, w0, states01[0]], axis=1), 10.0)
            output, states = self.lstm2(lstm2Input, states02)
            states02 = states
            # Eq. 17: the output layer sees a skip connection from every hidden layer
            outputs.append(tf.concat([states00[0], states01[0], states02[0]], axis=1))

        final = tf.stack(outputs, axis=1)
        final = self.mdn(final)
        final = clip_gradient(final, 100.0)
        pi, mux, muy, sigmax, sigmay, rho, penup = tf.split(final, [20,20,20,20,20,20,1], axis=2)

        # Eq. 18: e_t = 1 / (1 + exp(e_hat_t)), i.e. sigmoid(-e_hat_t)
        return tf.nn.softmax(pi), mux, muy, tf.exp(sigmax), tf.exp(sigmay), tf.nn.tanh(rho), 1.0 / (1.0 + tf.exp(penup)), pointsInput._keras_mask

def runTrainingLoop(model, optimizer, epochStart=1, batchStart=1):
    for epoch in range(epochStart, EPOCHS + 1):
        for i, batch in enumerate(tData):
            if epoch == epochStart and i < batchStart:
                continue
            start = time.time()
            points, text = batch
            with tf.GradientTape() as tape:
                a, b, c, d, e, f, g, mask = model(batch)
                lossNum = loss(a, b, c, d, e, f, g, points, mask)
            gradients = tape.gradient(lossNum, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # sampling + plotting is only for visual debugging, not training itself --
            # only pay for it (and the disk write) every 100 batches, not every batch
            if i%100 == 0:
                modelPointPreds = []
                for j in range(BATCH_SIZE):
                    real_len = int(np.sum(~np.all(points[j].numpy() == 999., axis=-1)))
                    modelPointPreds.append([])
                    for t in range(real_len):
                        modelPointPreds[j].append(samplePoint(a,b,c,d,e,f,g,timestep=t,sample=j))

                fig, axes = plt.subplots(2, 4, figsize=(20,8))
                for k in range(BATCH_SIZE):
                    visualizeStrokes(modelPointPreds[k], label=text[k], norms=datasetNorms, plott=axes[k//4,k%4])
                plt.tight_layout()
                plt.savefig(f"{GRAPH_PATH}/epoch{epoch}batch{i}.png")
                plt.close(fig)

            end = time.time()
            print(f"Epoch: {epoch}/{EPOCHS}, Batch: {i+1}/{NUM_BATCHES}, Loss: {lossNum.numpy()}, Time for batch: {round(end-start)}")
            if (i+1)%100 == 0 and i != 0:
                model.save_weights(f'{CHECKPOINT_PATH}/epoch_{epoch}batch_{i+1}.weights.h5')


def trainFromScratch():
    # Eqs. 38-45: rho=0.95 (n), momentum=0.9, lr=0.0001, epsilon=0.0001, centered -> uses (n - g^2)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4, rho=0.95, momentum=0.9, epsilon=1e-4, centered=True)
    model = HandwritingSynthesisModel()
    print("Model Initialized")
    runTrainingLoop(model, optimizer)


def trainFromCheckpoint(weightsPath):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4, rho=0.95, momentum=0.9, epsilon=1e-4, centered=True)
    model = HandwritingSynthesisModel()
    # layer shapes are only known after a real forward pass, so build the model
    # before load_weights can restore into it (same pattern as Testing.py)
    for points, text in tData.take(1):
        model((points, text))
    model.load_weights(weightsPath)
    epoch = int(weightsPath.split("_")[1].split("b")[0])
    batch = int(weightsPath.split("_")[-1].split(".")[0])
    print(f"Model Initialized, loaded weights from {weightsPath}")
    # restarting at the beginning of the data, even if the checkpoint was saved
    # partway through an epoch -- optimizer state (RMSprop accumulators) is fresh
    runTrainingLoop(model, optimizer, epochStart=epoch, batchStart=batch)


if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))

    trainFromCheckpoint(f'{CHECKPOINT_PATH}/epoch_5batch_1100.weights.h5')
    # trainFromScratch()