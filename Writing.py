import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

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
    decode = {".":"", "sp":" ", "ga":".", "km":",", "pt":"'", "sc":";"}
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



def visualizeStrokes(line, label=None):
    """
    line is a list of 3-long tuples containing the point data for the strokes that make up a line,
    displays a graph showing the different strokes that make up this line.
    label shows what the handwritten text is meant to display
    """
    x = [0]
    y = [0]
    for point in line:
        x.append(x[-1] + point[0])
        y.append(y[-1] + point[1])
        if point[2] == 1:
            x.pop(0)
            y.pop(0)
            plt.plot(x, y)
            x = [x[-1]]
            y = [y[-1]]
    if label != None:
        plt.title(label)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.show()

def createDataset(split):
    """
    split is a path to one of the three splits: training, validation, or testing
    returns a list if tuples (line strokes, label), line strokes are a list of tuples themselves
    """
    directories = []
    dataset = []
    labels = createLabelsDict()
    mainpath = "Dataset/Strokes/lineStrokes/"
    with open(split, "r") as f:
        for line in f:
            directories.append(line.strip())
    for folder in directories:
        pathnum = folder if len(folder) == 7 else folder[:-1]
        path = mainpath + folder.split("-")[0] + "/" + pathnum + "/"
        letter = "" if len(folder) == 7 else folder[-1]
        i = 1
        while True:
            try:
                fullpathnum = pathnum+letter+"-"+(str(i) if i>=10 else ("0"+str(i)))
            #    print(path + fullpathnum + ".xml")
                dataset.append((extractStrokeSequence(path + fullpathnum + ".xml"), labels[fullpathnum]))                    
                i+=1
            except FileNotFoundError:
                break
    return dataset

def computeDatasetMeanSTD(dataset, normalize=False, normalizeParams=None):
    """
    returns a tuple of the (meanX, meanY, stdX, stdY) of the given dataset
    if normalize = True, then also normalizes the dataset
    """
    x = []
    y = []
    if normalizeParams == None:
        for line, label in dataset:
            for point in line:
                x.append(point[0])
                y.append(point[1])
        normalizeParams = np.mean(x), np.mean(y), np.std(x), np.std(y)
    if normalize:
        for line, label in dataset:
            for i, point in enumerate(line):
                line[i] = ((point[0] - normalizeParams[0]) / normalizeParams[2],
                           (point[1] - normalizeParams[1]) / normalizeParams[3],
                           point[2])
    return normalizeParams

#visualizeStrokes(extractStrokeSequence(path), createLabelsDict()[pathName])
datasetTrain = createDataset("Dataset/trainset.txt")
datasetVal = createDataset("Dataset/testset_v.txt")
datasetTest = createDataset("Dataset/testset_t.txt")

datasetNorms = computeDatasetMeanSTD(datasetTrain, normalize=True)
computeDatasetMeanSTD(datasetVal, normalize=True, normalizeParams=datasetNorms)
computeDatasetMeanSTD(datasetTest, normalize=True, normalizeParams=datasetNorms)
print(datasetNorms)
# print(len(datasetT))
# for i, j in dataset:
#     print(j)