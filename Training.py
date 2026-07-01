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

        for timestep in range(tf.shape(pointsInput)[1].numpy()):
          #  print(timestep)
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

        return tf.nn.softmax(pi), mux, muy, tf.exp(sigmax) + 0.01, tf.exp(sigmay) + 0.01, tf.nn.tanh(rho), tf.nn.sigmoid(penup), pointsInput._keras_mask

if __name__ == "__main__":
    # for point, text in tData.take(1):
    #     model = HandwritingSynthesisModel()
    #     model((point, text))
    print(tf.config.list_physical_devices('GPU'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model = HandwritingSynthesisModel()
    print("Model Initialized")
    for epoch in range(EPOCHS):
        for i, batch in enumerate(tData):
            start = time.time()
            with tf.GradientTape() as tape:
                points, text = batch
                #print(points.shape, text.shape)
                a,b,c,d,e,f,g,mask = model(batch)
                lossNum = loss(a,b,c,d,e,f,g,points,mask)
            
       #     visualizeHeatmap(a[0,0,:],b[0,0,:],c[0,0,:],d[0,0,:],e[0,0,:],f[0,0,:],show=False,name=f"epoch{epoch}batch{i}")

            modelPointPreds = []
            for j in range(BATCH_SIZE):
                real_len = int(np.sum(~np.all(points[j].numpy() == 999., axis=-1)))
                modelPointPreds.append([])
                for t in range(real_len):
                    #visualizeSample(points[0], text[0], a, b, c, d, e, f, timestep=t, norms=datasetNorms)
                    modelPointPreds[j].append(samplePoint(a,b,c,d,e,f,g,timestep=t,sample=j))
            fig, axes = plt.subplots(2, 4, figsize=(20,8))
            for k in range(BATCH_SIZE):
                visualizeStrokes(modelPointPreds[k], label=text[k], norms=datasetNorms, plott=axes[k//4,k%4])
            plt.tight_layout()
            plt.savefig(f"{GRAPH_PATH}/epoch{epoch}batch{i}.png")
            plt.close(fig)

            gradients = tape.gradient(lossNum, model.trainable_variables)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
            end = time.time()
            print(f"Epoch: {epoch+1}/{EPOCHS}, Batch: {i+1}/{NUM_BATCHES}, Loss: {lossNum.numpy()}, Time for batch: {round(end-start)}")
            if i%10 == 0 and i != 0:
                model.save_weights(f'{CHECKPOINT_PATH}/epoch_{epoch}batch_{i}.weights.h5')