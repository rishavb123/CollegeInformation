import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
    
    clf = LinearRegression(n_jobs = 1)
    xs, ys = unpack(data)
    clf.fit(np.array(xs).reshape(-1, 1), ys)

    pred_x = 2020
    pred_y = clf.predict(pred_x)[0]
    print("In " + str(pred_x) + " the Average SAT Score of University of Michigan - Ann Arbor will be near " + str(pred_y))

    pred_xs = xs + [pred_x]
    pred_ys = [clf.predict(x) for x in pred_xs]

    plt.scatter(xs, ys, c='red')
    plt.plot(pred_xs, pred_ys, c='blue')
    plt.scatter(pred_x, pred_y)

    plt.show()
