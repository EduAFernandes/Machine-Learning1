import numpy as np
import pandas as pd
import math
from MLfunctions import generate_data, get_weighted_sum, update_bias, update_weights, cross_entropy, sigmoid
import matplotlib.pyplot


bias = 0.5
l_rate = 0.08
epochs = 40
epoch_loss = []

data, weights = generate_data(4, 3)


def train_model(data, weights, bias, l_rate, epochs):
    for e in range(epochs):
        individual_loss = []

        for i in range(len(data)):
            feature = data.loc[i][:-1]
            target = data.loc[i][-1]
            w_sum = get_weighted_sum(feature, weights, bias)
            prediction = sigmoid(w_sum)
            loss = cross_entropy(target, prediction)
            individual_loss.append(loss)
            # gradient descent
            weights = update_weights(weights, l_rate, target, prediction, feature)
            bias = update_bias(bias, l_rate, target, prediction)
        average_loss = sum(individual_loss)/len(individual_loss)
        epoch_loss.append(average_loss)
        print("-----------------------------------")
        print("epoch", e)
        print(average_loss)


train_model(data, weights, bias, l_rate, epochs)


# ploting the fucker
df = pd.DataFrame(epoch_loss)
df_plot = df.plot(kind="line", grid=True).get_figure()
df_plot.savefig("Training_Loss.pdf")
