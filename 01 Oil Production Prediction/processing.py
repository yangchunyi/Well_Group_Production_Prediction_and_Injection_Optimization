import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from setting import *

def merge_align(list_water, list_oil, min_date, max_date):
    # Merge water and oil production data with time alignment
    data_water = pd.concat(
        [pd.merge(list_water[i], on='prod_date', how='outer').fillna(0) for i in range(len(list_water))])
    data_water['prod_date'] = pd.to_datetime(data_water['prod_date'], format="%Y-%m-%d")
    data_water = data_water[(data_water['prod_date'] >= min_date) & (data_water['prod_date'] <= max_date)]

    data_oil = pd.concat([pd.merge(list_oil[i], on='prod_date', how='outer').fillna(0) for i in range(len(list_oil))])
    data_oil['prod_date'] = pd.to_datetime(data_oil['prod_date'], format="%Y-%m-%d")
    data_oil = data_oil[(data_oil['prod_date'] >= min_date) & (data_oil['prod_date'] <= max_date)]

    data_oil_water = pd.merge(data_water, data_oil, on='prod_date', how='left').fillna(0)
    data_oil_water.to_excel(f'{group_id}_all_data.xlsx')
    return data_oil_water


# Sliding window mechanism
def slip_window(data_oil_water, window_size_water, window_size_oil):
    scaler_x1 = MinMaxScaler(feature_range=(0, 1))
    scaler_x2 = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    xdata_in = data_oil_water.iloc[:, 1:-1]
    ydata = data_oil_water.iloc[window_size_water:, -1]  # Labels

    in_re_water, in_re_oil, out_re = [], [], []

    xdata_water = xdata_in.iloc[:-window_size_oil, :-out_dim]
    xdata_oil = xdata_in.iloc[window_size_water:, in_dim:]
    xdata_water = pd.DataFrame(scaler_x1.fit_transform(np.array(xdata_water)))
    xdata_oil = pd.DataFrame(scaler_x2.fit_transform(np.array(xdata_oil)))

    # Save max and min values
    max_values1, min_values1 = scaler_x1.data_max_, scaler_x1.data_min_
    max_values2, min_values2 = scaler_x2.data_max_, scaler_x2.data_min_

    with open(f'{group_id}_{window_size_oil}_min_max_values1.txt', 'w') as file:
        for i in range(len(max_values1)):
            file.write(f"Feature {i + 1} - max: {max_values1[i]}, min: {min_values1[i]}\n")
    with open(f'{group_id}_{window_size_oil}_min_max_values2.txt', 'w') as file:
        for i in range(len(max_values2)):
            file.write(f"Feature {i + 1} - max: {max_values2[i]}, min: {min_values2[i]}\n")

    for i in xdata_water.rolling(window_size_water):
        if len(i) == window_size_water:
            in_re_water.append(i)

    for i in xdata_oil.rolling(window_size_oil):
        if len(i) == window_size_oil:
            in_re_oil.append(i)

    for i in ydata.rolling(window_size_oil):
        if len(i) == window_size_oil:
            out_re.append(i.sum())

    # Remove anomalous samples
    index_list = [i for i in range(len(out_re)) if out_re[i] != 0]
    new_in_re_water = [in_re_water[i] for i in index_list]
    new_in_re_oil = [in_re_oil[i] for i in index_list]
    new_out_re = [out_re[i] for i in index_list]
    print('Number of anomalous samples removed:', len(out_re) - len(index_list))

    in_re_water = np.array(new_in_re_water).reshape(-1, window_size_water, in_dim)
    in_re_oil = np.array(new_in_re_oil).reshape(-1, window_size_oil, out_dim)
    out_re = np.array(new_out_re).reshape(-1, 1)

    out_re = scaler_y.fit_transform(out_re)

    return in_re_water, in_re_oil, out_re