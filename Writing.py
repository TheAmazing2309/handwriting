import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

path = "Dataset/Strokes/lineStrokes/a02/a02-000/a02-000-00.xml"
pathName = "a02-000-00"
labels = {}

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

#print(extractStrokeSequence(path))
#print(createLabelsDict())

visualizeStrokes(extractStrokeSequence(path), createLabelsDict()[pathName])