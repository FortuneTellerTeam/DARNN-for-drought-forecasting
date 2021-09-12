from flask import Flask, request, url_for, redirect, render_template, jsonify, send_file
import pandas as pd
import pickle
import numpy as np
from func import *
import torch
from torch import nn
import pandas as pd
import random
from torch.utils import data
import time
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import torch.optim.lr_scheduler as lr_scheduler
import json

from IPython.display import set_matplotlib_formats
# %matplotlib inline
from IPython import display

import glob
from PIL import Image


app = Flask(__name__)

# model = load_model('deployment_28042020')
# cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']


def predict_data(fips, L_3_4, L_4_3, date_forecast):

    batch_size = 128
    num_steps = 3
    num_hiddens_1 = 256
    num_hiddens_2 = 256
    lr = 0.001
    num_exoc = 18

    feature_list = ['PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX',
                    'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN',
                    'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']

    label_list = ['class']

    # loading the model
    print(f"{num_steps}_{num_hiddens_1}_{lr}_{batch_size}")
    params = torch.load(f'models/DARNN_{num_steps}_{num_hiddens_1}_{lr}_{batch_size}_lms.params')
    net = RNNModelScratch(num_exoc, num_hiddens_1, num_hiddens_2,
                          try_gpu(), params, init_lstm_state, lstm, False)
    print(f"The model is successfully loaded and the number of weight sets are length of parameters: {len(params)}")

    test_set = pd.read_pickle("data/test_data.pkl")

    data_valid = test_set.loc[(fips)]

    cleanup_nums = {"class": {"None": -1, "D0": -0.6,
                              "D1": -0.2, "D2": 0.2, "D3": 0.6, "D4": 1}}
    data_valid.replace(cleanup_nums, inplace=True)

    print(f"the date is {date_forecast}, {type(str(date_forecast))}")

    data_valid = data_valid[data_valid.index < str(date_forecast)]

    my_seq1, my_seq2 = get_time_series_inputs(data_valid, feature_list, L=len(
        data_valid), index_start=0), get_time_series_inputs(data_valid, label_list, L=len(data_valid), index_start=0)
    my_seq1 = my_seq1.T
    my_seq2 = my_seq2.T
    device = try_gpu()
    batch_size = 1
    test_iter = load_data_time_series(my_seq1, my_seq2, batch_size, num_steps,
                                      use_random_iter=False)
    optim_thr = threshold_set(
        L_3_4, L_4_3, p_d_3=0.054, p_d_4=0.021, sigma=0.1)
    print(f"We are about to start prediction with threshold {optim_thr}")

    predict_test(data_valid, net, test_iter, device, optim_thr)


@app.route('/')
def home():
    print("We are at home!")
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
    print("We are at prediction mode")

    int_features = [x for x in request.form.values()]
    print(f"int_features: {int_features}")
    final = np.array(int_features)
    print(f"final: {final}")

    fips = int(final[0])
    date_forecast = final[1]
    loss_ratio = float(final[2])

    print(f"The fips: {fips}, date: {date_forecast}, loss ratio: {loss_ratio}")
    # # the loss value in dollar$
    L_3_4 = loss_ratio
    L_4_3 = 1
    predict_data(fips, L_3_4, L_4_3, date_forecast)

    print('Forecast is finished!!')

    # # # filepaths
    fp_in = "figs/foo_*.png"
    fp_out = "static/image.gif"

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)

    return render_template('home.html', uploaded_image="image.gif")


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
