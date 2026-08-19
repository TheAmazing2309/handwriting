from Preprocessing import (tData, vData, fData, datasetNorms,
                           POINT_PAD_TOKEN, TEXT_PAD_TOKEN, VOCABSIZE, MAX_POINT_SEQ_LEN, MAX_TEXT_SEQ_LEN, CHECKPOINT_PATH,
                        BATCH_SIZE, GRAPH_PATH, visualizeSample, samplePoint, visualizeStrokes, encodeLine)
from Loss import loss
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
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

    def step(self, point, w0, h0, c0, h1, c1, h2, c2, kappa, textOneHot, textValidMask, u):
        """
        One recurrence timestep: advances all three LSTM layers and the attention window
        given the current (dx, dy, penup) input point. Shared by call() (teacher-forced,
        whole known sequences, batched) and generateHandwriting() (autoregressive, one
        sample at a time, no ground-truth points available) so the two don't drift apart.
        Returns the pre-MDN hidden concat -- mdnParams() turns that into distribution params.
        """
        pointWindow = clip_gradient(tf.concat([point, w0], 1), 10.0)
        _, (h0, c0) = self.lstm0(pointWindow, [h0, c0])
        alphaHat, betaHat, kappaHat = tf.split(self.windowDense(h0), 3, axis=1)
        kappa = kappa + tf.exp(kappaHat)
        alpha = tf.exp(alphaHat)
        beta = tf.exp(betaHat)
        phi = tf.reshape(tf.reduce_sum(tf.exp(-tf.reshape(beta, (-1,WINDOW_NUM,1)) * (tf.reshape(kappa, (-1,WINDOW_NUM,1)) - u) ** 2) * tf.reshape(alpha, (-1,WINDOW_NUM,1)), axis=1), (-1,MAX_TEXT_SEQ_LEN,1))
        phi = phi * tf.expand_dims(textValidMask, -1)
        w0 = tf.reduce_sum(phi * textOneHot, axis=1)
        lstm1Input = clip_gradient(tf.concat([point, w0, h0], axis=1), 10.0)
        _, (h1, c1) = self.lstm1(lstm1Input, [h1, c1])
        lstm2Input = clip_gradient(tf.concat([point, w0, h1], axis=1), 10.0)
        _, (h2, c2) = self.lstm2(lstm2Input, [h2, c2])
        hidden = tf.concat([h0, h1, h2], axis=1)
        return hidden, w0, h0, c0, h1, c1, h2, c2, kappa, phi

    def mdnParams(self, hidden):
        final = self.mdn(hidden)
        final = clip_gradient(final, 100.0)
        pi, mux, muy, sigmax, sigmay, rho, penup = tf.split(final, [20,20,20,20,20,20,1], axis=-1)
        return tf.nn.softmax(pi), mux, muy, tf.exp(sigmax), tf.exp(sigmay), tf.nn.tanh(rho), 1.0 / (1.0 + tf.exp(penup))

    def call(self, inputs, training=False):
        pointsInput, textInput = inputs
        batchSize = tf.shape(pointsInput)[0]
        seqLen = tf.shape(pointsInput)[1]

        textValidMask = tf.cast(tf.not_equal(textInput, TEXT_PAD_TOKEN), tf.float32)
        textOneHot = tf.one_hot(textInput, VOCABSIZE)
        pointsInput = self.pointsMask(pointsInput)
        textOneHot = self.textMask(textOneHot)

        w0Init = tf.zeros((batchSize, VOCABSIZE))
        zerosH = tf.zeros((batchSize, HIDDEN_SIZE))
        kappaInit = tf.zeros((batchSize, WINDOW_NUM))
        u = tf.reshape(tf.range(MAX_TEXT_SEQ_LEN, dtype=tf.float32), (1,1,-1))
        outputsInit = tf.TensorArray(dtype=tf.float32, size=seqLen, element_shape=(None, 3 * HIDDEN_SIZE))

        def cond(t, *_):
            return t < seqLen

        def body(t, w0, h0, c0, h1, c1, h2, c2, kappa, outputs):
            point = pointsInput[:, t, :]
            hidden, w0, h0, c0, h1, c1, h2, c2, kappa, _phi = self.step(
                point, w0, h0, c0, h1, c1, h2, c2, kappa, textOneHot, textValidMask, u)
            outputs = outputs.write(t, hidden)
            return t + 1, w0, h0, c0, h1, c1, h2, c2, kappa, outputs

        *_, outputs = tf.while_loop(
            cond, body,
            loop_vars=(tf.constant(0), w0Init, zerosH, zerosH, zerosH, zerosH, zerosH, zerosH, kappaInit, outputsInit)
        )

        final = tf.transpose(outputs.stack(), [1, 0, 2])
        pi, mux, muy, sigmax, sigmay, rho, penup = self.mdnParams(final)
        return pi, mux, muy, sigmax, sigmay, rho, penup, pointsInput._keras_mask

def makeTrainStep(model, optimizer):
    @tf.function
    def trainStep(points, text):
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


VAL_CHECK_BATCHES = 3

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
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4, rho=0.95, momentum=0.9, epsilon=1e-4, centered=True)
    model = HandwritingSynthesisModel()
    print("Model Initialized")
    runTrainingLoop(model, optimizer)


def trainFromCheckpoint(weightsPath):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4, rho=0.95, momentum=0.9, epsilon=1e-4, centered=True)
    model = HandwritingSynthesisModel()
    for points, text in tData.take(1):
        model((points, text))
    model.load_weights(weightsPath)
    epoch = int(weightsPath.split("_")[1].split("b")[0])
    batch = int(weightsPath.split("_")[-1].split(".")[0])
    print(f"Model Initialized, loaded weights from {weightsPath}")
    runTrainingLoop(model, optimizer, epochStart=epoch, batchStart=batch)


def latestCheckpoint():
    ckptRe = re.compile(r"epoch_(\d+)batch_(\d+)\.weights\.h5$")
    ckpts = []
    for path in glob.glob(f"{CHECKPOINT_PATH}/*.weights.h5"):
        m = ckptRe.search(path)
        if m:
            ckpts.append((int(m.group(1)), int(m.group(2)), path))
    return max(ckpts)[2]


def generateHandwriting(model, text, maxSteps=700):
    """
    Autoregressively synthesizes a stroke sequence for arbitrary text that was never in the
    dataset -- there's no ground-truth stroke sequence to teacher-force on, so each step
    samples a point from the MDN and feeds it back in as the next input, unlike call() which
    is fed real points throughout. Returns a list of (dx, dy, penup) tuples in the same format
    visualizeStrokes already consumes.
    Stops once the attention window has swept past the last real character (Graves 2013,
    Sec. 5.3: stop the first time phi(t, U) > phi(t, u) for all u < U), or after maxSteps as
    a safety cap. kappa only ever increases, so this can't trigger prematurely.
    """
    encoded = encodeLine(text)
    realLen = len(encoded)
    encoded = encoded[:MAX_TEXT_SEQ_LEN] + [TEXT_PAD_TOKEN] * max(0, MAX_TEXT_SEQ_LEN - len(encoded))
    textInput = tf.constant([encoded], dtype=tf.int32)
    textValidMask = tf.cast(tf.not_equal(textInput, TEXT_PAD_TOKEN), tf.float32)
    textOneHot = tf.one_hot(textInput, VOCABSIZE)
    u = tf.reshape(tf.range(MAX_TEXT_SEQ_LEN, dtype=tf.float32), (1, 1, -1))

    point = tf.zeros((1, 3))
    w0 = tf.zeros((1, VOCABSIZE))
    h0 = c0 = h1 = c1 = h2 = c2 = tf.zeros((1, HIDDEN_SIZE))
    kappa = tf.zeros((1, WINDOW_NUM))

    points = []
    for _ in range(maxSteps):
        hidden, w0, h0, c0, h1, c1, h2, c2, kappa, phi = model.step(
            point, w0, h0, c0, h1, c1, h2, c2, kappa, textOneHot, textValidMask, u)
        pi, mux, muy, sigmax, sigmay, rho, penup = model.mdnParams(hidden)
        dx, dy, penState = samplePoint(
            tf.expand_dims(pi, 1), tf.expand_dims(mux, 1), tf.expand_dims(muy, 1),
            tf.expand_dims(sigmax, 1), tf.expand_dims(sigmay, 1), tf.expand_dims(rho, 1),
            tf.expand_dims(penup, 1), timestep=0, sample=0)
        points.append((dx, dy, penState))
        point = tf.constant([[dx, dy, float(penState)]], dtype=tf.float32)

        if int(tf.argmax(phi[0, :, 0])) >= realLen - 1:
            break
    return points


if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))

    trainFromCheckpoint(latestCheckpoint())
    # trainFromScratch()
