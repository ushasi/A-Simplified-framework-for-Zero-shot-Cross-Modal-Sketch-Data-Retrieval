from gensim.test.utils import datapath, get_tmpfile, common_texts
from gensim.models import KeyedVectors, Word2Vec
import scipy.io
import numpy as np
import datetime

def load_word2vec_dataset():

         
    words = []
    words.append("airplane")
    words.append("alarm clock")
    words.append("ant")
    words.append("ape")
    words.append("apple")
    words.append("metal")#armour
    words.append("axe")
    words.append("banana")
    words.append("bat")
    words.append("bear")
    words.append("bee")
    words.append("beetle")
    words.append("bell")
    words.append("bench")
    words.append("bicycle")
    words.append("blimp")
    words.append("bread")
    words.append("butterfly")
    words.append("cabin")
    words.append("camel")
    words.append("candle")
    words.append("cannon")
    words.append("car")
    words.append("castle")
    words.append("cat")
    words.append("chair")
    words.append("chicken")
    words.append("church")
    words.append("couch")
    words.append("cow")
    words.append("crab")
    words.append("crocodile")
    words.append("cup")
    words.append("deer")
    words.append("dog")
    words.append("dolphin")
    words.append("door")
    words.append("duck")
    words.append("elephant")
    words.append("eyeglasses")
    words.append("fan")
    words.append("fish")
    words.append("flower")
    words.append("frog")
    words.append("geyser")
    words.append("giraffe")
    words.append("guitar")
    words.append("hamburger")
    words.append("hammer")
    words.append("harp")
    words.append("hat")
    words.append("hedgehog")
    words.append("helicopter")
    words.append("hermit crab")
    words.append("horse")
    words.append("hot air balloon")
    words.append("hot dog")
    words.append("hour glass")
    words.append("jack o lantern")
    words.append("jelly fish")
    words.append("kangaroo")
    words.append("knife")
    words.append("lion")
    words.append("lizard")
    words.append("lobster")
    words.append("motorcycle")
    words.append("mouse")
    words.append("mushroom")
    words.append("owl")
    words.append("parrot")
    words.append("pear")
    words.append("penguin")
    words.append("piano")
    words.append("pickup truck")
    words.append("pig")
    words.append("pineapple")
    words.append("pistol")
    words.append("pizza")
    words.append("pretzel")
    words.append("Rabbit")
    words.append("raccoon")
    words.append("racket")
    words.append("ray")
    words.append("rhinoceros")
    words.append("rifle")
    words.append("rocket")
    words.append("sail boat")
    words.append("saw")
    words.append("saxophone")
    words.append("scissors")
    words.append("scorpion")
    words.append("seagull")
    words.append("seal")
    words.append("sea turtle")
    words.append("shark")
    words.append("sheep")
    words.append("shoe")
    words.append("skyscraper")
    words.append("snail")
    words.append("snake")
    words.append("songbird")
    words.append("spider")
    words.append("spoon")
    words.append("squirrel")
    words.append("starfish")
    words.append("strawberry")
    words.append("swan")
    words.append("sword")
    words.append("table")
    words.append("tank")
    words.append("teapot")
    words.append("teddy bear")
    words.append("tiger")
    words.append("tree")
    words.append("trumpet")
    words.append("turtle")
    words.append("umbrella")
    words.append("violin")
    words.append("volcano")
    words.append("wading bird")
    words.append("wheel chair")
    words.append("windmill")
    words.append("window")
    words.append("wine bottle")
    words.append("zebra")



    model = KeyedVectors.load_word2vec_format('dataset/GoogleNews-vectors-negative300.bin', binary=True) 	
    wv_embeddings = np.zeros((125,300))
    #print(model['cars'])
    #print type(model['cars'])

    for i in range(125):
        if i == 1:
                wv_embeddings[i,:] = (model['alarm'] + model['clock'])/2
        elif i == 6:
                wv_embeddings[i,:] = model['metal']
        elif i == 53:
                wv_embeddings[i,:] = (model['hermit'] + model['crab'])/2
        elif i == 55:
                wv_embeddings[i,:] = (model['hot'] + model['air'] + model['balloon'])/3
        elif i == 56:
                wv_embeddings[i,:] = (model['hot'] + model['dog'])/2
        elif i == 57:
                wv_embeddings[i,:] = (model['hour'] + model['glass'])/2
        elif i == 58:
                wv_embeddings[i,:] = (model['jack'] + model['lantern'])/2
        elif i == 59:
                wv_embeddings[i,:] = (model['jelly'] + model['fish'])/2
        elif i == 73:
                wv_embeddings[i,:] = (model['pickup'] + model['truck'])/2
        elif i == 86:
                wv_embeddings[i,:] = (model['sail'] + model['boat'])/2
        elif i == 93:
                wv_embeddings[i,:] = (model['sea'] + model['turtle'])/2
        elif i == 111:
                wv_embeddings[i,:] = (model['teddy'] + model['bear'])/2
        elif i == 119:
                wv_embeddings[i,:] = (model['wading'] + model['bird'])/2
        elif i == 120:
                wv_embeddings[i,:] = (model['wheel'] + model['chair'])/2
        elif i == 123:
                wv_embeddings[i,:] = (model['wine'] + model['bottle'])/2
        else:
                print(i)
                wv_embeddings[i,:] = model[words[i]]
        scipy.io.savemat('dataset/wv_embeddings.mat', {'features':wv_embeddings}) #saving


   
    return words

h = load_word2vec_dataset()
