import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os, time

#MAX_STROKE_LEN = float('inf')
MAX_STROKE_LEN = 300
MAX_POINT_SEQ_LEN = 1940
#MAX_POINT_SEQ_LEN = 5
MAX_TEXT_SEQ_LEN = 71
BATCH_SIZE = 1
ON_COLAB = os.path.exists('/content')
SAVE_PATH = '/content/drive/MyDrive/handwriting/Processed/' if ON_COLAB else 'Processed/'
USE_SAVED_NP_DATA = False
SAVE_NEW_DATA_TO_NP = False

charToIndex = {"[PAD]":0, "[SOS]":1, "[EOS]":2}
indexToChar = {0:"[PAD]", 1:"[SOS]", 2:"[EOS]"}
decode = {".":"", "sp":" ", "ga":".", "km":",", "pt":"'", "sc":";"}
with open("Dataset/letters", "r") as f:
    for i, line in enumerate(f):
        char = line.strip() if len(line) <= 2 else decode[line.strip()]
        charToIndex[char] = i + 3
        indexToChar[i+3] = char

VOCABSIZE = len(charToIndex)
POINT_PAD_TOKEN = (999,999,999)
TEXT_PAD_TOKEN = 0

def encodeLine(line):
    """
    takes a str (line) and returns an array of numbers that corresponds to the given string with added SOS and EOS tokens
    """
    out = [1]
    for i in line:
        out.append(charToIndex[i])
    out.append(2)
    return out

def decodeLine(line):
    out = ""
    line = line[1:-1]
    for i in line:
        out += indexToChar[i] if i >2 else ""
    return out

def extractStrokeSequence(path):
    """
    Takes a path (str) and returns a list of tuples corresponding to the points in that line
    (dx, dy, penup) -> dx, dy are relative differences from the previous point, penup is 1 if the
        pen is lifted, 0 otherwise.
    """
    tree = ET.parse(path)
    strokeset = tree.getroot()[1]
    line = [(0,0,0)]
    #xOffset, yOffset = -int(strokeset[0][0].attrib['x']), -int(strokeset[0][0].attrib['y'])
    for i, stroke in enumerate(strokeset):
        if i != 0:
            line.append((int(stroke[0].attrib['x']) - int(strokeset[i-1][-1].attrib['x']), 
                        int(stroke[0].attrib['y']) - int(strokeset[i-1][-1].attrib['y']),
                        0))
        for j in range(1, len(stroke)):
            line.append((int(stroke[j].attrib['x']) - int(stroke[j-1].attrib['x']), 
                        int(stroke[j].attrib['y']) - int(stroke[j-1].attrib['y']),
                        1 if j == len(stroke) - 1 else 0))
    return line

def createLabelsDict():
    """
    Returns a dict (str,str) of all lines in the labels file with line number as the key and matching text as the value. 
    """
    labels = {}
    with open("Dataset/labels.mlf", "r") as f:
        key = ""
        value = ""
        for line in f:
            line = line.strip()
            if line[0] == '"':
                if key != "":
                    labels[key] = value
                key = line.split("/")[-1][:-5]
                value = ""
            elif line == '.' or (line.isalnum() and len(line) > 1):
                value += decode[line]
            elif len(line) <= 2:
                value += line
        labels[key] = value
    return labels

labels = createLabelsDict()

def visualizeStrokes(line, label=None, plott=None, norms=None):
    """
    line is a list of 3-long tuples containing the point data for the strokes that make up a line,
    displays a graph showing the different strokes that make up this line.
    label shows what the handwritten text is meant to display
    norms: if provided as (mean_x, mean_y, std_x, std_y), denormalizes dx/dy before plotting
    """
    x = [0]
    y = [0]
    line = np.array(line, dtype=np.float32)
    if norms is not None:
        mean_x, mean_y, std_x, std_y = norms
        pad_mask = np.all(line == 999., axis=-1)
        line[~pad_mask, 0] = line[~pad_mask, 0] * std_x + mean_x
        line[~pad_mask, 1] = line[~pad_mask, 1] * std_y + mean_y
    line = list(line)
    for point in line:
        point = tuple(point)
        if np.array_equal(point, POINT_PAD_TOKEN):
            break
        x.append(x[-1] + point[0])
        y.append(y[-1] + point[1])
        if point[2] == 1:
            x.pop(0)
            y.pop(0)
            if plott is None:
                plt.plot(x, y)
            else:
                plott.plot(x,y)
            x = [x[-1]]
            y = [y[-1]]
    if len(x) > 1:
        if plott is None:
            plt.plot(x, y)
        else:
            plott.plot(x,y)
    if label != None:
        if plott is None:
            plt.title(label)
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')
            plt.show()
        else:
            plott.set_title(label)
            plott.invert_yaxis()
    

# def visualizeHeatmap(pi, mux, muy, sigmax, sigmay, rho, show=False, name=None):
#     mux = tf.reshape(mux, (20,1,1))
#     muy = tf.reshape(muy, (20,1,1))
#     sigmax = tf.reshape(sigmax, (20,1,1))
#     sigmay = tf.reshape(sigmay, (20,1,1))
#     rho = tf.reshape(rho, (20,1,1))
#     pi = tf.reshape(pi, (20,1,1))
#     x = np.linspace(-3, 3, 100) 
#     y = np.linspace(-3, 3, 100)
#     epsilon = 1e-6
#     X, Y = np.meshgrid(x, y)
#     X = tf.cast(tf.reshape(X, (1, 100, 100)), dtype=tf.float32)
#     Y = tf.cast(tf.reshape(Y, (1, 100, 100)), dtype=tf.float32)
#     Z = ((((X-mux)/sigmax)**2)+(((Y-muy)/sigmay)**2))-(2*rho*(X-mux)*(Y-muy)/(sigmax*sigmay))
#     N = tf.exp(-Z/(2*(1-rho**2+epsilon))) / (2*3.14159*sigmax*sigmay*(1-rho**2+epsilon)**0.5)
#     plt.contourf(x, y, tf.reduce_sum(N*pi, axis=0))
#     plt.colorbar()
#     if show: plt.show()
#     else:
#         plt.savefig("Graphs/"+
#                     str(time.time()) if name == None else str(name)
#                     +".png")
#         plt.close()

def visualizeSample(points, text, pi, mux, muy, sigmax, sigmay, rho, timestep=0, norms=None):
    """
    points, text: single sample (not batched)
    pi, mux, muy, sigmax, sigmay, rho: full model outputs shaped (batch, seq_len, 20)
    timestep: which output timestep to visualize
    norms: (mean_x, mean_y, std_x, std_y) used to denorm mixture to pixel space
    """
    fig, (upper, lower) = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
    fig.patch.set_facecolor('white')
    lower.set_facecolor('white')

    visualizeStrokes(points.numpy(), label=decodeLine(text.numpy()), plott=upper, norms=norms)

    mean_x, mean_y, std_x, std_y = norms if norms is not None else (0, 0, 1, 1)

    pts = points.numpy()
    pad_mask = np.all(pts == 999., axis=-1)
    pts_d = pts.copy()
    pts_d[~pad_mask, 0] = pts_d[~pad_mask, 0] * std_x + mean_x
    pts_d[~pad_mask, 1] = pts_d[~pad_mask, 1] * std_y + mean_y
    valid = ~pad_mask[:timestep + 1]
    abs_x = float(pts_d[:timestep + 1][valid, 0].sum())
    abs_y = float(pts_d[:timestep + 1][valid, 1].sum())

    mux_d  = mux[0, timestep, :].numpy()  * std_x + mean_x + abs_x
    muy_d  = muy[0, timestep, :].numpy()  * std_y + mean_y + abs_y
    sx_d   = np.maximum(sigmax[0, timestep, :].numpy() * std_x, 1e-2)
    sy_d   = np.maximum(sigmay[0, timestep, :].numpy() * std_y, 1e-2)
    rho_v  = rho[0, timestep, :].numpy()
    pi_v   = pi[0, timestep, :].numpy()

    xlim = upper.get_xlim()
    ylim = upper.get_ylim()
    xg = np.linspace(min(xlim), max(xlim), 300)
    yg = np.linspace(min(ylim), max(ylim), 300)
    X, Y = np.meshgrid(xg, yg)

    pi_t  = pi_v.reshape(20, 1, 1)
    mux_t = mux_d.reshape(20, 1, 1)
    muy_t = muy_d.reshape(20, 1, 1)
    sx_t  = sx_d.reshape(20, 1, 1)
    sy_t  = sy_d.reshape(20, 1, 1)
    rho_t = rho_v.reshape(20, 1, 1)
    Xg    = X[np.newaxis]
    Yg    = Y[np.newaxis]

    eps = 1e-6
    Z = (((Xg - mux_t) / sx_t)**2 + ((Yg - muy_t) / sy_t)**2
         - 2 * rho_t * (Xg - mux_t) * (Yg - muy_t) / (sx_t * sy_t))
    density = np.sum(pi_t * np.exp(-Z / (2 * (1 - rho_t**2 + eps)))
                     / (2 * np.pi * sx_t * sy_t * (1 - rho_t**2 + eps)**0.5), axis=0)

    lower.contourf(xg, yg, density, levels=20, cmap='Blues', alpha=0.4)
    lower.set_xlim(xlim)
    lower.set_ylim(ylim)   # preserves inversion to match upper
    lower.set_title(f"Predicted next-point distribution (t={timestep})")

    plt.tight_layout()
    plt.show()


def createDataset(split):
    """
    split is a path to one of the three splits: training, validation, or testing
    returns a tuple of lists (line strokes, label), line strokes are a list of tuples themselves
    """
    directories = []
    datasetPoints = []
    datasetText = []
    mainpath = "Dataset/Strokes/lineStrokes/"
    with open(split, "r") as f:
        for line in f:
            directories.append(line.strip())
    j = 0
    for folder in directories:
        pathnum = folder if len(folder) == 7 else folder[:-1]
        path = mainpath + folder.split("-")[0] + "/" + pathnum + "/"
        letter = "" if len(folder) == 7 else folder[-1]
        i = 1
        while True:
            try:
                fullpathnum = pathnum+letter+"-"+(str(i) if i>=10 else ("0"+str(i)))
            #    print(path + fullpathnum + ".xml")
                strokes = extractStrokeSequence(path + fullpathnum + ".xml")
                if len(strokes) <= MAX_STROKE_LEN:
                    datasetPoints.append(strokes)    
                    datasetText.append(encodeLine(labels[fullpathnum]))               
                i+=1
                j+=1
                print("Read xml file ",j,"out of 9000ish")
            except FileNotFoundError:
                break
    return datasetPoints, datasetText

def computeDatasetMeanSTD(datasetPoints, normalize=False, normalizeParams=None):
    """
    returns a tuple of the (meanX, meanY, stdX, stdY) of the given dataset POINTS LIST ONLY
    if normalize = True, then also normalizes the dataset
    """
    x = []
    y = []
    if normalizeParams == None:
        for line in datasetPoints:
            for point in line:
                x.append(point[0])
                y.append(point[1])
        normalizeParams = np.mean(x), np.mean(y), np.std(x), np.std(y)
    if normalize:
        for line in datasetPoints:
            for i, point in enumerate(line):
                line[i] = ((point[0] - normalizeParams[0]) / normalizeParams[2],
                           (point[1] - normalizeParams[1]) / normalizeParams[3],
                           point[2])
    return normalizeParams

def generator(dataset):
    """
    Generator function to be used by tf to convert python list-based data into a tf.data.Dataset
    """
    for feature, label in dataset:
        yield feature, label

def toTFDataset(dataset):
    """
    Wrapper to convert python list-based dataset to a tf.data.Dataset, uses generator
    """
    return tf.data.Dataset.from_generator(
        lambda: generator(dataset),
        output_signature=(
            tf.TensorSpec(shape=(min(MAX_POINT_SEQ_LEN, MAX_STROKE_LEN), 3), dtype=tf.float32),
            tf.TensorSpec(shape=(MAX_TEXT_SEQ_LEN,), dtype=tf.int32)
        )
    ).batch(BATCH_SIZE)

# #visualizeStrokes(extractStrokeSequence(path), createLabelsDict()[pathName])
if not os.path.exists(SAVE_PATH+"datasetTrainPoints.npy") or not USE_SAVED_NP_DATA:
    print("LOADING DATASETS FROM XML!")
    datasetTrainPoints, datasetTrainText = createDataset("Dataset/trainset.txt")
    datasetTrainPointsExtra, datasetTrainTextExtra = createDataset("Dataset/testset_f.txt")
    datasetTrainPoints.extend(datasetTrainPointsExtra)
    datasetTrainText.extend(datasetTrainTextExtra)
    datasetValPoints, datasetValText = createDataset("Dataset/testset_v.txt")
    datasetTestPoints, datasetTestText = createDataset("Dataset/testset_t.txt")

    datasetNorms = computeDatasetMeanSTD(datasetTrainPoints, normalize=True)
    computeDatasetMeanSTD(datasetValPoints, normalize=True, normalizeParams=datasetNorms)
    computeDatasetMeanSTD(datasetTestPoints, normalize=True, normalizeParams=datasetNorms)

    datasetTrainPoints = tf.keras.utils.pad_sequences(datasetTrainPoints, maxlen=min(MAX_POINT_SEQ_LEN, MAX_STROKE_LEN), padding='post', value=POINT_PAD_TOKEN, dtype='float32')
    datasetValPoints = tf.keras.utils.pad_sequences(datasetValPoints, maxlen=min(MAX_POINT_SEQ_LEN, MAX_STROKE_LEN), padding='post', value=POINT_PAD_TOKEN, dtype='float32')
    datasetTestPoints = tf.keras.utils.pad_sequences(datasetTestPoints, maxlen=min(MAX_POINT_SEQ_LEN, MAX_STROKE_LEN), padding='post', value=POINT_PAD_TOKEN, dtype='float32')
    datasetTrainText = tf.keras.utils.pad_sequences(datasetTrainText, maxlen=MAX_TEXT_SEQ_LEN, padding='post', value=TEXT_PAD_TOKEN)
    datasetValText = tf.keras.utils.pad_sequences(datasetValText, maxlen=MAX_TEXT_SEQ_LEN, padding='post', value=TEXT_PAD_TOKEN)
    datasetTestText = tf.keras.utils.pad_sequences(datasetTestText, maxlen=MAX_TEXT_SEQ_LEN, padding='post', value=TEXT_PAD_TOKEN)

    if SAVE_NEW_DATA_TO_NP:
        np.save(SAVE_PATH+"datasetTrainPoints.npy", datasetTrainPoints)
        np.save(SAVE_PATH+"datasetTrainText.npy", datasetTrainText)
        np.save(SAVE_PATH+"datasetValPoints.npy", datasetValPoints)
        np.save(SAVE_PATH+"datasetValText.npy", datasetValText)
        np.save(SAVE_PATH+"datasetTestPoints.npy", datasetTestPoints)
        np.save(SAVE_PATH+"datasetTestText.npy", datasetTestText)
        np.save(SAVE_PATH+"datasetNorms.npy", datasetNorms)

else:
    print("LOADING DATASETS FROM NPY!")
    datasetTrainPoints = np.load(SAVE_PATH+"datasetTrainPoints.npy")
    datasetTrainText = np.load(SAVE_PATH+"/datasetTrainText.npy")
    datasetValPoints = np.load(SAVE_PATH+"datasetValPoints.npy")
    datasetValText = np.load(SAVE_PATH+"datasetValText.npy")
    datasetTestPoints = np.load(SAVE_PATH+"datasetTestPoints.npy")
    datasetTestText = np.load(SAVE_PATH+"datasetTestText.npy")
    datasetNorms = tuple(np.load(SAVE_PATH+"datasetNorms.npy"))

tData = toTFDataset(list(zip(datasetTrainPoints, datasetTrainText)))
vData = toTFDataset(list(zip(datasetValPoints, datasetValText)))
fData = toTFDataset(list(zip(datasetTestPoints, datasetTestText)))


if __name__ == "__main__":
    # for feature, label in tData.take(1):
    #     print("Feature: ", feature)
    #     print("Label: ", label)
    
    # i = 0
    # av = 0
    # for feature, label in tData:
    #     i+=1
    #     av+=feature.shape[0]
    #     print(feature.shape[0])

    # lenFeatures = [len(f) for f, l in tData]
    # print("_", len([len(f) for f, l in tData]), "max", max([len(f) for f, l in tData]), "maxTextLen", max([len(l) for f, l in tData]))
    # print("v", len([len(f) for f, l in vData]), "max", max([len(f) for f, l in vData]), "maxTextLen", max([len(l) for f, l in vData]))
    # print("f0", len([len(f) for f, l in fData]), "max", max([len(f) for f, l in fData]), "maxTextLen", max([len(l) for f, l in fData]))

    # print("Total examples: ", len(lenFeatures))
    # print("Average length of point sequence: ", sum(lenFeatures)/len(lenFeatures))
    # print("50 percentile: ", np.percentile(lenFeatures, 50))
    # print("75 percentile: ", np.percentile(lenFeatures, 75))
    # print("90 percentile: ", np.percentile(lenFeatures, 90))
    # print("95 percentile: ", np.percentile(lenFeatures, 95))
    # print("Max value: ", max(lenFeatures))

    for f, l in fData.take(1):
        visualizeStrokes(f[0].numpy(), label=decodeLine(l[0].numpy()), norms=datasetNorms)