from Preprocessing import (tData, vData, fData, datasetNorms, 
                           POINT_PAD_TOKEN, TEXT_PAD_TOKEN, VOCABSIZE, MAX_POINT_SEQ_LEN, MAX_TEXT_SEQ_LEN, CHECKPOINT_PATH,
                        BATCH_SIZE, GRAPH_PATH, visualizeSample, samplePoint, visualizeStrokes)
from Loss import loss
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # headless: plots are only ever saved to disk, never shown interactively
import matplotlib.pyplot as plt
import numpy as np
import time
import glob
import re

WINDOW_NUM = 10
HIDDEN_SIZE = 400
PREDS_NUM = 20
NUM_BATCHES = sum(1 for _ in tData)

EPOCHS = 200


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
        seqLen = tf.shape(pointsInput)[1]

        # true (non-[PAD]) character positions, so the window never attends to padding
        textValidMask = tf.cast(tf.not_equal(textInput, TEXT_PAD_TOKEN), tf.float32)
        textInput = tf.one_hot(textInput, VOCABSIZE)
        pointsInput = self.pointsMask(pointsInput)
        textInput = self.textMask(textInput)

        w0Init = tf.zeros((batchSize, VOCABSIZE))
        zerosH = tf.zeros((batchSize, HIDDEN_SIZE))
        kappaInit = tf.zeros((batchSize, WINDOW_NUM))
        u = tf.reshape(tf.range(MAX_TEXT_SEQ_LEN, dtype=tf.float32), (1,1,-1))
        outputsInit = tf.TensorArray(dtype=tf.float32, size=seqLen, element_shape=(None, 3 * HIDDEN_SIZE))

        # Explicit tf.while_loop (rather than a Python for-loop over tf.range) so this dynamic-length
        # recurrence traces once regardless of AutoGraph's handling of Keras Model.call() internals --
        # AutoGraph does not reliably convert plain Python for-loops written inside call().
        def cond(t, *_):
            return t < seqLen

        def body(t, w0, h0, c0, h1, c1, h2, c2, kappa, outputs):
            point = pointsInput[:, t, :] #shape(batch, 3)
            pointWindow = clip_gradient(tf.concat([point, w0], 1), 10.0) #shape(batch,VOCABSIZE+3)
            _, (h0, c0) = self.lstm0(pointWindow, [h0, c0]) #[hidden,cell]
            alphaHat, betaHat, kappaHat = tf.split(self.windowDense(h0), 3, axis=1)
            kappa = kappa + tf.exp(kappaHat)
            alpha = tf.exp(alphaHat)
            beta = tf.exp(betaHat)
            phi = tf.reshape(tf.reduce_sum(tf.exp(-tf.reshape(beta, (batchSize,WINDOW_NUM,1)) * (tf.reshape(kappa, (batchSize,WINDOW_NUM,1)) - u) ** 2) * tf.reshape(alpha, (batchSize,WINDOW_NUM,1)), axis=1), (batchSize,-1,1))
            phi = phi * tf.expand_dims(textValidMask, -1) # never attend to [PAD] characters
            w0 = tf.reduce_sum(phi * textInput, axis=1)
            lstm1Input = clip_gradient(tf.concat([point, w0, h0], axis=1), 10.0)
            _, (h1, c1) = self.lstm1(lstm1Input, [h1, c1])
            lstm2Input = clip_gradient(tf.concat([point, w0, h1], axis=1), 10.0)
            _, (h2, c2) = self.lstm2(lstm2Input, [h2, c2])
            # Eq. 17: the output layer sees a skip connection from every hidden layer
            outputs = outputs.write(t, tf.concat([h0, h1, h2], axis=1))
            return t + 1, w0, h0, c0, h1, c1, h2, c2, kappa, outputs

        *_, outputs = tf.while_loop(
            cond, body,
            loop_vars=(tf.constant(0), w0Init, zerosH, zerosH, zerosH, zerosH, zerosH, zerosH, kappaInit, outputsInit)
        )

        final = tf.transpose(outputs.stack(), [1, 0, 2])
        final = self.mdn(final)
        final = clip_gradient(final, 100.0)
        pi, mux, muy, sigmax, sigmay, rho, penup = tf.split(final, [20,20,20,20,20,20,1], axis=2)

        # Eq. 18: e_t = 1 / (1 + exp(e_hat_t)), i.e. sigmoid(-e_hat_t)
        return tf.nn.softmax(pi), mux, muy, tf.exp(sigmax), tf.exp(sigmay), tf.nn.tanh(rho), 1.0 / (1.0 + tf.exp(penup)), pointsInput._keras_mask

def makeTrainStep(model, optimizer):
    @tf.function
    def trainStep(points, text):
        # batches are padded to the dataset-wide max length (MAX_POINT_SEQ_LEN), but most
        # sequences are far shorter -- trimming to this batch's own real max length keeps
        # the recurrence from unrolling over pure padding. This slice is data-dependent
        # (not shape-dependent), so it lives inside the traced graph without forcing a
        # retrace: the input signature stays (BATCH_SIZE, MAX_POINT_SEQ_LEN, 3) every call.
        validMask = tf.reduce_any(tf.not_equal(points, POINT_PAD_TOKEN[0]), axis=-1)
        realLen = tf.maximum(tf.reduce_max(tf.reduce_sum(tf.cast(validMask, tf.int32), axis=1)), 1)
        points = points[:, :realLen, :]
        with tf.GradientTape() as tape:
            pi, mux, muy, sigmax, sigmay, rho, penup, mask = model((points, text), training=True)
            lossNum = loss(pi, mux, muy, sigmax, sigmay, rho, penup, points, mask)
        gradients = tape.gradient(lossNum, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return lossNum, pi, mux, muy, sigmax, sigmay, rho, penup, points
    return trainStep


VAL_CHECK_BATCHES = 3  # not the whole validation set -- just enough for a quick generalization check

def makeEvalStep(model):
    @tf.function
    def evalStep(points, text):
        validMask = tf.reduce_any(tf.not_equal(points, POINT_PAD_TOKEN[0]), axis=-1)
        realLen = tf.maximum(tf.reduce_max(tf.reduce_sum(tf.cast(validMask, tf.int32), axis=1)), 1)
        points = points[:, :realLen, :]
        pi, mux, muy, sigmax, sigmay, rho, penup, mask = model((points, text), training=False)
        return loss(pi, mux, muy, sigmax, sigmay, rho, penup, points, mask)
    return evalStep


def computeValLoss(evalStep):
    losses = [float(evalStep(points, text)) for points, text in vData.take(VAL_CHECK_BATCHES)]
    return sum(losses) / len(losses)


def runTrainingLoop(model, optimizer, epochStart=1, batchStart=1):
    trainStep = makeTrainStep(model, optimizer)
    evalStep = makeEvalStep(model)
    for epoch in range(epochStart, EPOCHS + 1):
        for i, batch in enumerate(tData):
            if epoch == epochStart and i < batchStart:
                continue
            start = time.time()
            rawPoints, text = batch
            lossNum, a, b, c, d, e, f, g, points = trainStep(rawPoints, text)

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
                valLoss = computeValLoss(evalStep)
                print(f"  -> Validation loss: {valLoss}")
                with open("ValidationMetrics.txt", "a") as f:
                    f.write(f"Epoch: {epoch}/{EPOCHS}, Batch: {i+1}/{NUM_BATCHES}, ValLoss: {valLoss}\n")


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


def latestCheckpoint():
    ckptRe = re.compile(r"epoch_(\d+)batch_(\d+)\.weights\.h5$")
    ckpts = []
    for path in glob.glob(f"{CHECKPOINT_PATH}/*.weights.h5"):
        m = ckptRe.search(path)
        if m:
            ckpts.append((int(m.group(1)), int(m.group(2)), path))
    return max(ckpts)[2]


if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))

    trainFromCheckpoint(latestCheckpoint())
    # trainFromScratch()