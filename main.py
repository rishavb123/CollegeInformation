import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from neural_network import NeuralNetwork
import numpy as np

def unpack(data):
    xs, ys = [], []
    for point in data:
        xs.append(point[0])
        ys.append(point[1])
    return xs, ys

with open('response.json') as json_file:
    full_data = json.load(json_file)['results'][0]
    
    data = []
    for i in range(2001, 2018):
        data.append([i, full_data[str(i)]['admissions']['sat_scores']['average']['overall']])
    
    xs, ys = unpack(data)
    
    lr = LinearRegression(n_jobs = 1)
    lr.fit(np.array(xs).reshape(-1, 1), ys)

    pred_x = 2020
    pred_xs = xs + [pred_x]

    lr_pred_y = lr.predict(pred_x)[0]
    print("Linear Regression: In " + str(pred_x) + " the Average SAT Score of University of Michigan - Ann Arbor will be near " + str(lr_pred_y))

    lr_pred_ys = [lr.predict(x) for x in pred_xs]

    nn = NeuralNetwork(1, 2, 1)
    epoch = 1000000
    nn.train([[(x - 2001) / 17] for x in xs], [[y / 1600] for y in ys], epoch)

    nn_pred_y = nn.predict([(pred_x - 2001) / 17])[0] * 1600
    print("Neural Network: In " + str(pred_x) + " the Average SAT Score of University of Michigan - Ann Arbor will be near " + str(nn_pred_y))

    nn_pred_ys = [nn.predict([(x - 2001) / 17])[0] * 1600 for x in pred_xs]

    nn.delete()

    plt.scatter(xs, ys, c='green')

    plt.plot(pred_xs, lr_pred_ys, c='blue')
    plt.scatter(pred_x, lr_pred_y, c='purple')

    plt.plot(pred_xs, nn_pred_ys, c='red')
    plt.scatter(pred_x, nn_pred_y, c='orange')

    plt.show()
