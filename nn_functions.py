import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import random
from sklearn.metrics import accuracy_score

def sigma(x):
    s = 1/(1 + np.exp(-x))
    return s


def sigma_der(x):
    s = np.exp(-x)/(1+np.exp(-x))**2
    return s


def relu(x):
    return np.maximum(0, x)


def relu_der(x):
    x = np.asarray(x)
    return (x > 0).astype(x.dtype)


def det_activations(inputs, weights, biases, relu_act):
    
    activations = []
    activations.append(np.dot(weights[0],inputs) + biases[0])
    
    for i in range(1,len(biases)):

        activations.append(np.dot(weights[i],activations[i-1]) + biases[i])

    if relu_act:
        activations = [relu(act) for act in activations]
        activations = [np.round(act, 3) for act in activations]

        return activations

    else:
        activations = [sigma(act) for act in activations]
        activations = [np.round(act, 3) for act in activations]
            
        return activations


def grad(act, wt, bs, ins, outs, relu_act):
    
    delta = [0*a for a in act]
    gr_bs = [0*b for b in bs]
    gr_wt = [0*w for w in wt]

    L = len(act)

    if relu_act:
        delta[-1] = 2*(act[-1] - outs) * relu_der(np.dot(wt[-1], act[-2]) + bs[-1])

        for i in range(2,L):
            delta[-i] = np.dot(wt[-i+1].T, delta[-i+1]) * relu_der(np.dot(wt[-i], act[-i-1]) + bs[-i])

        delta[-L] = np.dot(wt[-L+1].T, delta[-L+1]) * relu_der(np.dot(wt[-L], ins) + bs[-L])
    
    else:
        delta[-1] = 2*(act[-1] - outs) * sigma_der(np.dot(wt[-1], act[-2]) + bs[-1])

        for i in range(2,L):
            delta[-i] = np.dot(wt[-i+1].T, delta[-i+1]) * sigma_der(np.dot(wt[-i], act[-i-1]) + bs[-i])

        delta[-L] = np.dot(wt[-L+1].T, delta[-L+1]) * sigma_der(np.dot(wt[-L], ins) + bs[-L])        

    gr_bs = delta

    for j in range(len(act[0])):
        for k in range(len(ins)):
            gr_wt[0][j][k] = ins[k] * delta[0][j]
    
    for l in range(1,L):
        for j in range(len(act[l])):
            for k in range(len(act[l-1])):
                gr_wt[l][j][k] = act[l-1][k] * delta[l][j]
                
    
    return gr_wt, gr_bs


def neural_network(inputs, outputs, nodes, learning_rate, epochs, batch_size, relu_act, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)
  
    nodes.append(outputs.shape[1])
    nodes.insert(0, inputs.shape[1])
    
    biases = [np.random.randn(y) for y in nodes[1:]]
    weights = [np.random.randn(x, y) for x, y in zip(nodes[1:], nodes[:-1])]

    cost = []
    count = 0

    for k in range(epochs):

        data = np.hstack((inputs, outputs))
        np.random.shuffle(data)
        data_part = [data[m:m + batch_size] for m in range(0, len(data), batch_size)]

        for l in range(len(data_part)):

            inputs_part = data_part[l][:,:inputs.shape[1]]
            outputs_part = data_part[l][:,inputs.shape[1]:]
            
            iter_cost = 0
    
            gr_b = [0*bias for bias in biases]
            gr_w = [0*weight for weight in weights]
        
            for j in range(len(inputs_part)):                
            
                activations = det_activations(inputs_part[j], weights, biases, relu_act)
                iter_cost +=  (1/len(inputs_part)) * sum([(activations[-1].tolist()[i] - outputs_part[j][i])**2 for i in range(len(outputs_part[j]))])
            
                gr_w_p, gr_b_p = grad(activations, weights, biases, inputs_part[j], outputs_part[j], relu_act)
    
                gr_b = [g + (1/len(inputs_part))*gp for g, gp in zip(gr_b, gr_b_p)]
                gr_w = [g + (1/len(inputs_part))*gp for g, gp in zip(gr_w, gr_w_p)]
                
            weights = [w - learning_rate * gw for w, gw in zip(weights, gr_w)]
            biases = [b - learning_rate * gb for b, gb in zip(biases, gr_b)]

            count += 1
            
            if count % 10 == 0:
                cost.append(iter_cost)

    return weights, biases, cost


def nn_plot(inputs_train, outputs_train, inputs_test, outputs_test, labels, nodes, learning_rate, epochs, batch_size, relu_act=False, random_state=None):

    start = time.time()  
    weights, biases, cost = neural_network(inputs_train, outputs_train, nodes, learning_rate, epochs, batch_size, relu_act, random_state) 
    end = time.time()
    
    print("Time taken:", np.round(end - start, 2), "seconds")
    
    plt.plot(cost);
    plt.grid(alpha=0.6);
    
    outputs_predict = predict(inputs_test, weights, biases, labels, relu_act)
    print('Accuracy: '+ str(accuracy_score(outputs_test,outputs_predict)))

    return weights, biases, cost


def predict(inputs, weights, biases, labels, relu_act):

    outputs = []

    for i in range(len(inputs)):

        activations = det_activations(inputs[i], weights, biases, relu_act)
        outputs.append(labels[np.argmax(activations[-1])])

    return outputs