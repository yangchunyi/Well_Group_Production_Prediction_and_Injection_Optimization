import pickle
from sklearn.metrics import r2_score
from setting import *
from traing import *
from processing import *

if __name__ == '__main__':

    for i in range(in_num):
        data_in = pd.read_excel(
            f'{path}{group_id}/inject/{water_wellid[i]}.xlsx', usecols=water_features)[water_features]
        if i == 0:
            merged_df = pd.read_excel(
                f'{path}{group_id}/inject/{water_wellid[0]}.xlsx', usecols=water_features)[water_features]
        else:
            merged_df = pd.concat([merged_df, data_in], ignore_index=True)

        list_water.append(data_in)

    merged_df = merged_df.drop_duplicates()
    merged_df = merged_df.sort_values(by='prod_date')
    merged_df = merged_df.set_index('prod_date')
    min_date = merged_df.index.min()
    max_date = merged_df.index.max()
    print('min_date:', min_date)
    print('max_date:', max_date)
    for i in range(len(oil_wellid)):
        data_out = pd.read_excel(
            f'{path}{group_id}/output/{oil_wellid[i]}.xlsx', usecols=oil_features)[oil_features]
        list_oil.append(data_out)

    data_allout = pd.read_excel(
        f'{path}{group_id}/output/SumDailyOil.xlsx',
        usecols=label_features)[label_features]
    list_oil.append(data_allout)
    metric_mae_all = []
    metric_val_mae_all = []
    metric_mse_all = []
    auc_list = []
    data_water_oil = merge_align(list_water, list_oil, min_date, max_date)
    print(data_water_oil.columns)
    for window_size_oil in window_sizes:
        # window_size_oil：The sliding window size of the production well
        xdata_in_water, xdata_in_oil, ydata_out = slip_window(data_water_oil, window_size_water, window_size_oil)
        sample_num = xdata_in_water.shape[0]
        test_num = int(test_ratio * sample_num)
        xdata1, xdata2, ydata = xdata_in_water[:sample_num - test_num], xdata_in_oil[:sample_num - test_num], ydata_out[
                                                                                                              :sample_num - test_num]
        test_xdata1, test_xdata2, test_ydata = xdata_in_water[-test_num:], xdata_in_oil[-test_num:], ydata_out[
                                                                                                     -test_num:]
        shape1 = [sample_num, window_size_water, in_dim]
        shape2 = [sample_num, window_size_oil, out_dim]

        metric_mae, metric_val_mae, metric_mse, count, model = run_model(xdata1, xdata2, ydata, shape1, shape2, Epochs,
                                                                         Batch_Size,
                                                                         Learning_rate)
        pre_data = model.predict([test_xdata1, test_xdata2])

        with open(str(group_id) + '_' + str(window_size_oil) + '_scaler_y.pkl', 'rb') as f:
            loaded_scaler_y = pickle.load(f)
        pre_data = loaded_scaler_y.inverse_transform(pre_data)
        test_ydata = loaded_scaler_y.inverse_transform(test_ydata)
        # Plot the difference between true and predicted values
        test_ydata = test_ydata.reshape(-1, )
        pre_data = pre_data.reshape(-1, )
        plot_picture_pre(test_ydata, pre_data)
        pre_data = np.array(pre_data).reshape(-1, )
        test_ydata = np.array(test_ydata).reshape(-1, )
        # 计算R²值
        r2 = r2_score(pre_data, test_ydata)
        print(f'R²: {r2:.4f}')

    best_i = 0
    for i in range(len(window_sizes)):
        if min(metric_val_mae_all[i]) < min(metric_val_mae_all[best_i]):
            best_i = i
    metric_mae = metric_mae_all[best_i]
    metric_mse = metric_mse_all[best_i]
    print('metric_mae:', metric_mae)
    print('metric_mse:', metric_mse)
    print(f'The optimal time window size of the producing well in well group {group_id} is {window_sizes[best_i]}')
