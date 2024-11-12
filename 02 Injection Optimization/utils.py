import numpy as np
from processing import *
from keras.models import load_model

def custom_min_max_scaler(data, max_vals, min_vals):
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

def objective_function(params):
    test_xdata1, test_xdata2 = make_sample(params)
    test_xdata1 = custom_min_max_scaler(test_xdata1, max_values1, min_values1)
    test_xdata2 = custom_min_max_scaler(test_xdata2, max_values2, min_values2)
    test_xdata1 = np.clip(test_xdata1, 0, 1)
    test_xdata2 = np.clip(test_xdata2, 0, 1)
    test_xdata1 = np.array(test_xdata1).reshape(1, window_size_water, in_dim)
    test_xdata2 = np.array(test_xdata2).reshape(1, window_size_oil, out_dim)
    best_model = load_model(
        str(group_id) + '_' + str(window_size_oil) + '_best_pre_model.h5')  # 替换为您的模型文件路径
    pre_ydata = best_model.predict([test_xdata1, test_xdata2])  # 替换为您的测试数据
    pre_ydata = np.clip(pre_ydata, 0, 1)
    pre_ydata = loaded_scaler_y.inverse_transform(pre_ydata)
    pre_ydata = pre_ydata.reshape(-1, )
    best_pre_ydata = pre_ydata[0]
    return pre_ydata[0]
