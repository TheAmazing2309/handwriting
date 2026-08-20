from Training import HandwritingSynthesisModel, generateHandwriting, latestCheckpoint
from Preprocessing import fData, visualizeSample, visualizeStrokes, datasetNorms, charToIndex, MAX_TEXT_SEQ_LEN
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Generate handwriting for text using the latest trained checkpoint")
parser.add_argument("text", nargs="?", default=None, help="Text to generate handwriting for")
args = parser.parse_args()

model = HandwritingSynthesisModel()
print("model init")
for points, text in fData.take(1):
    pi, mux, muy, sigmax, sigmay, rho, penup, mask = model((points, text))

print("first dummy sample run")
model.load_weights(latestCheckpoint())
print("model loaded")

for points, text in fData.take(1):
    pi, mux, muy, sigmax, sigmay, rho, penup, mask = model((points, text))
    print("second sample passed")
    for timestep in range(10):
        visualizeSample(points[0], text[0], pi, mux, muy, sigmax, sigmay, rho, timestep=timestep)

text = args.text if args.text is not None else input("Enter text to generate handwriting for: ")
unsupported = sorted(set(c for c in text if c not in charToIndex))
if unsupported:
    print(f"Unsupported characters (not in training vocabulary): {unsupported}")
else:
    if len(text) > MAX_TEXT_SEQ_LEN - 2:
        print(f"Text too long, truncating to {MAX_TEXT_SEQ_LEN - 2} characters")
        text = text[:MAX_TEXT_SEQ_LEN - 2]
    points = generateHandwriting(model, text)
    plt.figure()
    visualizeStrokes(points, label=text, norms=datasetNorms)
    outPath = f"Graphs/Generated_{text[:30].replace(' ', '_')}.png"
    plt.savefig(outPath)
    print(f"Saved to {outPath}")