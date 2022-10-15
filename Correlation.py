import numpy
import os

FEATURE_NORM_DIR = "Feature/genresNorm/"
FEATURES_LENGTH = 13

list1 = []
list2 = []

def getFeatures(a, b):
    list1 = []
    list2 = []
    for dir in os.listdir(FEATURE_NORM_DIR):
        folder = os.path.join(FEATURE_NORM_DIR, dir)
        for filename in os.listdir(folder):
            if filename.endswith("npy"):
                filename = os.path.join(folder, filename)
                narray = numpy.load(filename)
                list1.append(narray[a])
                list2.append(narray[b])
    return list1, list2

def writeToCSV():
    textfile = open('correlationFeaturesNorm.csv', 'a+')
    textfile.write('Features, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, y\n')
    for i in range(FEATURES_LENGTH):
        wstr = str(i)
        for j in range(FEATURES_LENGTH):
            list1, list2 = getFeatures(i, j)
            wstr += ', ' + str(numpy.corrcoef(list1, list2)[0, 1])
        listy = []
        for j in range(1000):
            listy.append(j % 100)
        wstr += ', ' + str(numpy.corrcoef(list1, listy)[0, 1])
        wstr += '\n'
        textfile.write(wstr)

writeToCSV()