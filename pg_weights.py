# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 22:16:05 2019

@author: Jakub
"""
w0 = 0
import numpy as np

ws = [w0]
alpha = 1

for i in range(0):
    w_old = ws[-1]
    p = np.exp(w_old) / (np.exp(w_old) + 1)
    print("p =", p)
    w_new = w_old + alpha *  p * (1 - p)
    ws.append(w_new)
    
    w_old = ws[-1]
    p = np.exp(w_old) / (np.exp(w_old) + 1)
    print("p =", p)
    w_new = w_old - alpha *  p * (1 - p)
    ws.append(w_new)

"""
print("PHASE 1")
for i in range(855):
    w_old = ws[-1]
    p = np.exp(w_old) / (np.exp(w_old) + 1)
    print("p =", p)
    w_new = w_old + alpha *  p * (1 - p)
    ws.append(w_new)

print(30*"-")
print("PHASE 2")  
  
for i in range(855+10):
    w_old = ws[-1]
    p = np.exp(w_old) / (np.exp(w_old) + 1)
    print("p =", p)
    w_new = w_old - alpha *  p * (1 - p)
    ws.append(w_new)
    
"""

def gradient(weights, rewards):
    g = np.zeros_like(weights)
    prob = np.exp(weights - np.max(weights)) / np.sum(np.exp(weights - np.max(weights)))
    for i in range(len(weights)):
        der = 0
        for j in range(len(weights)):
            if i == j:
                der += rewards[j] * prob[i] * (1-prob[i])
            else:                
                der -= rewards[j] * prob[i] * prob[j]
        g[i] = der
    
    return g

r1 = np.asarray([1, 0, 0.551])
r2 = np.asarray([0, 1, 0.551])
r3 = np.asarray([0, 1, 0])

w = np.zeros(3)
alpha = 8
for i in range(20):
    w += alpha * gradient(w, r1)
    print(w)
    w += alpha * gradient(w, r2)
    print(w)

    
        