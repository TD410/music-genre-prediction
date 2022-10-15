import matplotlib.pyplot as plt
import numpy
import os

FEATURES_DIR = "Feature/genres/"
FEATURES_LENGTH = 13
PLOT_DES = "Boxplot/"

features = [[] for i in range(FEATURES_LENGTH)]

def getFeatures():
    for dir in os.listdir(FEATURES_DIR):
        folder = os.path.join(FEATURES_DIR, dir)
        for filename in os.listdir(folder):
            if filename.endswith("npy"):
                filename = os.path.join(folder, filename)
                narray = numpy.load(filename)
                for i in range(FEATURES_LENGTH):
                    features[i].append(narray[i])
        print("Complete load data from " + folder)
    return features

def getFeaturesOfGenre(path):
    for filename in os.listdir(path):
        if filename.endswith("npy"):
            filename = os.path.join(path, filename)
            narray = numpy.load(filename)
            for i in range(FEATURES_LENGTH):
                features[i].append(narray[i])
    print("Complete load data from " + path)
    return features

def boxplotAll(data, path):
    plt.figure(figsize=(15,10))
    plt.boxplot(data)
    plt.savefig(path)
    plt.show()
    
def boxplotEachFeature(data, path):
    for i in range(FEATURES_LENGTH):
        plt.figure(figsize=(15,2))
        plt.boxplot(data[i], 0, 'o', 0)
        plt.savefig(path + "feature" + str(i + 1) + ".png")
        plt.show()
        
def detectOutliers(data):
    downbound = [(2.5 * numpy.percentile(data[i], 25) - 1.5 * numpy.percentile(data[i], 75)) for i in range(FEATURES_LENGTH)]
    upbound = [(2.5 * numpy.percentile(data[i], 75) - 1.5 * numpy.percentile(data[i], 25)) for i in range(FEATURES_LENGTH)]
    outliers = []
    for i in range(len(features[0])):
        outFeatures = []
        for j in range(FEATURES_LENGTH):
            if (features[j][i] < downbound[j] or features[j][i] > upbound[j]):
                outFeatures.append(j)
        if len(outFeatures) > 0:
            outliers.append([i, outFeatures])
    return outliers

def writeOutliersToFile(outliers, genre):
    textfile = open('Outliers.csv', 'a+')
    textfile.write('Outliers detected by IQR method\n')
    textfile.write('Genre, Song Number, Outlier Features, Reject\n')
    for item in outliers:
        reject = 0
        if len(item[1]) > 6:
            reject = 1
        textfile.write(genre + ',' + str(item[0]) + ',' + str(item[1]).replace(',', ';') + ',' + str(reject) + '\n')
    textfile.close()

for dir in os.listdir(FEATURES_DIR):
    folder = os.path.join(FEATURES_DIR, dir)
    features = getFeaturesOfGenre(folder)
    writeOutliersToFile(detectOutliers(features), dir)