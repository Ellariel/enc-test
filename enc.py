import networkx as nx
import random
import os, sys, time, pickle
from tqdm import tqdm, notebook

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model, load_model

import warnings
warnings.filterwarnings("ignore")

class Autoencoder(Model):
  def __init__(self, latent_dim, n_):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(n_, activation='sigmoid'),
      layers.Flatten()
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class Enc():
    def __init__(self, n, latent_dim=64):
        self.n = n
        self.autoencoder = Autoencoder(latent_dim, self.n) 
        self.autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError()) #MeanSquaredError() #CosineSimilarity()

    def train(self, x_train, x_test, seed=48):
        np.random.seed(seed)
        self.autoencoder.fit(x_train, x_train,
                        epochs=20,
                        shuffle=True,
                        validation_data=(x_test, x_test))
    
    def save(self, save_dir='./encoder'):
        self.autoencoder.decoder.save_weights(os.path.join(save_dir, 'saved_decoder'))
        self.autoencoder.encoder.save_weights(os.path.join(save_dir, 'saved_encoder'))
        
    def load(self, save_dir='./encoder'):
        self.autoencoder.decoder.load_weights(os.path.join(save_dir, 'saved_decoder'))
        self.autoencoder.encoder.load_weights(os.path.join(save_dir, 'saved_encoder'))
        
    def encode(self, obs):
        return self.autoencoder.encoder(tf.convert_to_tensor([obs])).numpy()

def get_shortest_path(u, v, proto='dijkstra'):
    try:
        path = nx.shortest_path(G, u, v, method=proto)
    except:
        return
    return path

if __name__ == '__main__':

    base_dir = './'
    snapshots_dir = os.path.join(base_dir, 'snapshots')
    weights_dir = os.path.join(base_dir, 'weights')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    prepared_graph = os.path.join(snapshots_dir, f'prepared_graph.pickle')
    #warmed_graph = os.path.join(snapshots_dir, f'warmed_graph.pickle')

    with open(prepared_graph, 'rb') as f:
        prepared = pickle.load(f)
        G = prepared['directed_graph']
        T = prepared['transactions']

    for i in T:
        if nx.shortest_path_length(G, i[0], i[1]) <= 1:
            print('!')

    id_to_idx = {}
    idx_to_id = {}
    for idx, id in enumerate(G.nodes):
        id_to_idx[id] = idx
        idx_to_id[idx] = id
        
    n = len(G.nodes)
    
    O = []
    for t in tqdm(T):
        obs = np.zeros((n, ), dtype=np.float32)
        p = get_shortest_path(t[0], t[1])
        for i in p:
            obs[id_to_idx[i]] = 10
        O.append(obs)   

    x_train, x_test = O[:6000], O[6000:]
    x_train = tf.convert_to_tensor(x_train) 
    x_test = tf.convert_to_tensor(x_test) 
       
    e = Enc(n)
    e.train(x_train, x_test)
    e.save()
