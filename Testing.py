from Training import HandwritingSynthesisModel
from Preprocessing import fData, visualizeSample
import tensorflow as tf

model = HandwritingSynthesisModel()
print("model init")
for points, text in fData.take(1):
    pi, mux, muy, sigmax, sigmay, rho, penup = model((points, text))

print("first dummy sample run")
model.load_weights("Checkpoints/epoch_0batch_25.weights.h5")
print("model loaded")

for points, text in fData.take(1):
    pi, mux, muy, sigmax, sigmay, rho, penup = model((points, text))
    print("second sample passed")
    for timestep in range(10):  # adjust range as needed
        visualizeSample(points[0], text[0], pi, mux, muy, sigmax, sigmay, rho, timestep=timestep)
