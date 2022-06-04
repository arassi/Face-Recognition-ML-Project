
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.decomposition import TruncatedSVD
import random,time

from cv2 import *
import cv2
import paho.mqtt.client as mqtt

import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from scipy.stats import mode

def on_connect(client, userdata, flags, rc):
    print("Connected to server (i.e., broker) with result code "+str(rc))

def on_message(client, userdata, msg):
    print("on_message: " + msg.topic + " " + str(msg.payload, "utf-8"))


def process_image(img):
    img = cv2.resize(img, (128, 128))
    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def load_images(wildcard, how_many=100):
    allfiles = list(glob(wildcard))

    N = how_many
    M = len(cv2.imread(allfiles[0],cv2.IMREAD_GRAYSCALE).flatten())
    matrix = np.zeros((M,N))

    imgshape = cv2.imread(allfiles[0], cv2.IMREAD_GRAYSCALE).shape

    allfiles = random.choices(allfiles, k=how_many)

    for i,f in enumerate(allfiles):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = img.flatten()
        matrix[:,i] = img
    print("--",imgshape)
    return matrix,imgshape            


def train():
    all5,_ = load_images("newdata/train/5R*.png", 30)
    all0,imgshape = load_images("newdata/train/0R*.png", 30)
    print("imgshape", imgshape)
    all_digits = np.zeros((all5.shape[0], all5.shape[1] + all0.shape[1]))

    all_digits[:,0:all5.shape[1]] = all5
    all_digits[:,all5.shape[1]:] = all0

    svd = TruncatedSVD()
    factors = svd.fit_transform(all_digits)

    fac1 = factors[:,0].reshape(imgshape)
    fac2 = factors[:,1].reshape(imgshape)

    return factors

def project2D(factors,flatimgs):
    return flatimgs.T @ factors

def savemodel(point=(None,None)):
    factors = train()
   
    sample5,_ = load_images("./newdata/train/5R*.png", 30)
    sample0,_ = load_images("./newdata/train/0R*.png", 30)

    p5 = project2D(factors, sample5)
    p0 = project2D(factors, sample0)
    


    df = pd.DataFrame()

    for i in range(p5.shape[0]):
        df = pd.concat([df, pd.DataFrame.from_records([{"f1":p5[i,0], "f2":p5[i,1], "class":5}])], ignore_index = True)


    for i in range(p0.shape[0]):
        df = pd.concat([df,pd.DataFrame.from_records([{"f1":p0[i,0], "f2":p0[i,1], "class":0}])], ignore_index = True)


    # plt.scatter(p5[:,0],p5[:,1], color="green")
    # plt.scatter(p5[:,0],p5[:,1], color="red")

    sns.scatterplot(data = df, x = "f1", y="f2", hue = "class")
    if point[0]==None: pass
    else: sns.scatterplot(x=[point[0]], y=[point[1]], color="green")
        
    plt.savefig("res.png")
    plt.clf()    
    # p0 = p0[p0 > 0]
    # p5 = p5[p5 > 0]
    
    model = {"p5":p5, "p0":p0, "factors":factors}
    
    import pickle
    pickle.dump(model,open("model.pkl",'wb'))
    
    return model
    
def distancecalc(p1,p2):
	distance = np.linalg.norm(p1 - p2)
	return distance


def KNN(pointsA, pointsB, pointTest,  k=5, verbose=True):
	distancesA = [ (distancecalc(pointTest, pA), 'A') for pA in pointsA ]
	distancesB = [ (distancecalc(pointTest, pB),'B') for pB in pointsB ]

	all_distances = distancesA + distancesB
	sorted_distances = sorted(all_distances, key= lambda y: y[0])
	top_k = sorted_distances[:k]
	top_k_classes = [ v[1] for v in top_k ]

	from collections import Counter
	counts = Counter(top_k_classes).items()
	voted_class = sorted(counts, key=lambda y: y[1], reverse=True)[0][0]
	# print(top_k_classes)
	# print(voted_class)

	if verbose: print(voted_class)
	
	return 0 if voted_class=='A' else 1
	

from signal import signal, SIGINT
from sys import exit

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)
	
if __name__ == "__main__":
    signal(SIGINT, handler)

    model = savemodel()
    p0 = model["p0"]
    p5 = model["p5"]
    factors = model["factors"]
        

    flatimages, size = load_images("./newdata/train/0R*.png",10)
    # print("fi-",flatimages.shape)
    test_points = project2D(factors, flatimages)
    print(test_points.shape)
    for rowidx in range(test_points.shape[0]):
        tp = test_points[rowidx,:]
        print(KNN(p0,p5,tp))
        
    
    client = mqtt.Client()
    client.on_message = on_message
    client.on_connect = on_connect
    client.connect(host="broker.emqx.io", port=1883, keepalive=60)
    #client.loop_start()
    camera = cv2.VideoCapture(0)
    num = 0

    while (True):
        num +=1

        print("Press enter to capture", end="")
        input()

        success,img = camera.read()    
        img = process_image(img)
        
        img_projected = project2D(factors, img.flatten().T)
        print(img_projected.shape)
        num = KNN(p0,p5,img_projected, verbose=True)
        savemodel(img_projected)

        if num == 0:
            client.publish("arassi/lcd", "Sampad")
        if num == 1:
            client.publish("arassi/lcd", "Bill")


        time.sleep(0.1)

    camera.release()
    del(camera)
