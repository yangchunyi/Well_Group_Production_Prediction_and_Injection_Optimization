import pickle
import time
from utils import *
from processing import *
from apso_clm import *
from datetime import timedelta
from setting import *

if __name__ == '__main__':
    merged_df = pd.read_excel(
        f'{path}{group_id}/inject/{water_wellid[0]}.xlsx', usecols=water_features)
    merged_df['prod_date'] = pd.to_datetime(merged_df['prod_date'], format="%Y-%m-%d")
    start_time3 = time.time()
    max_values1, max_values2 = [], []
    min_values1, min_values2 = [], []
    with open(str(group_id) + '_' + str(window_size_oil) + '_min_max_values1.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Feature" in line:
                parts = line.split(":")
                max_value = float(parts[1].split(",")[0].strip())
                min_value = float(parts[2].split(",")[0].strip())
                max_values1.append(max_value)
                min_values1.append(min_value)
    with open(str(group_id) + '_' + str(window_size_oil) + '_min_max_values2.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Feature" in line:
                parts = line.split(":")
                max_value = float(parts[1].split(",")[0].strip())
                min_value = float(parts[2].split(",")[0].strip())
                max_values2.append(max_value)
                min_values2.append(min_value)
    max_values1 = np.array(max_values1)
    min_values1 = np.array(min_values1)
    max_values2 = np.array(max_values2)
    min_values2 = np.array(min_values2)

    with open(str(group_id) + '_' + str(window_size_oil) + '_scaler_y.pkl', 'rb') as f:
        loaded_scaler_y = pickle.load(f)

    for i in range(len(water_wellid)):
        data_in = pd.read_excel(
            f'{path}{group_id}/inject/{water_wellid[i]}.xlsx', usecols=['prod_date', 'inj_duration', 'inj_daily'])
        data_in['prod_date'] = pd.to_datetime(data_in['prod_date'], format="%Y-%m-%d")
        merged_df = pd.concat([merged_df, data_in], ignore_index=True)
        condition = data_in['inj_duration'] == 24
        filtered_df = data_in[condition]
        filtered_df = np.array(filtered_df['inj_daily'])
        q1 = np.percentile(filtered_df, 25)
        q3 = np.percentile(filtered_df, 75)
        iqr = q3 - q1
        limit_lb = q1 - 1.5 * iqr
        limit_ub = q3 + 1.5 * iqr
        list_lb.append(limit_lb)
        list_ub.append(limit_ub)

    for i in range(in_num):
        if max_values1[2 * i] < list_ub[i]:
            list_ub[i] = max_values1[2 * i]
        if list_lb[i] <= 0:
            list_lb[i] = list_ub[i] * 1 / 4
        if min_values1[2 * i] > list_lb[i]:
            list_lb[i] = min_values1[2 * i]

    print('list_lb:', list_lb)
    print('list_ub:', list_ub)

    data_allout = pd.read_excel(
        f'{path}{group_id}/output/SumDailyOil.xlsx',
        usecols=label_features)
    data_allout['prod_date'] = pd.to_datetime(data_allout['prod_date'], format="%Y-%m-%d")
    tol_v = data_allout[data_allout['remark'] != 0]['sum_dailyoil'].mean() * 0.01 * window_size_oil
    tol_v = round(tol_v, 2)
    print('tol_v:', tol_v)
    merged_df = merged_df.sort_values(by='prod_date')
    merged_df = merged_df.set_index('prod_date')
    min_date = merged_df.index.min()
    max_date = merged_df.index.max()
    print('min_date:', min_date)
    print('max_date:', max_date)
    pre_date_min = max_date + timedelta(days=1 + shift_num)
    pre_date_max = max_date + timedelta(days=shift_num + window_size_oil)
    print('pre_date_min: ', pre_date_min.date())
    print('pre_date_max: ', pre_date_max.date())

    start_time1 = time.time()
    list_lb = np.array(list_lb)
    list_ub = np.array(list_ub)
    apso = Apso(list_lb,list_ub, w_max, w_min, c1, c2)
    best_params, best_pre_ydata = apso.forward(objective_function,swarmsize_v,maxiter_v,tol_v,patience_v)

    for i in range(len(best_params)):
        if best_params[i] < list_lb[i]:
            best_params[i] = list_lb[i]
        if best_params[i] > list_ub[i]:
            best_params[i] = list_ub[i]
    best_params = [round(value, 2) for value in best_params]
    print("best_params:", best_params)
    re_excel = pd.DataFrame(columns=['well_id', 'inj_daily'])
    for i in range(in_num):
        water_inj = best_params[i]
        now_water_wellid = water_wellid[i]
        select_data1 = relate_data[relate_data['well_id'] == now_water_wellid]
        new_data = pd.DataFrame({'well_id': [now_water_wellid], 'inj_daily': [water_inj], 'production_state_1': [0]})
        re_excel = pd.concat([re_excel, new_data], ignore_index=True)
    re_excel.to_excel('re_excel.xlsx')

    print("Predicted value：", round(best_pre_ydata, 2))
    test_xdata1, test_xdata2 = make_sample(best_params)
    test_xdata_merge = pd.concat([test_xdata1, test_xdata2], axis=1)
    test_xdata_merge.to_excel(str(group_id) + '_best_test_xdata.xlsx')
    print('pre_date_min: ', pre_date_min.date())
    print('pre_date_max: ', pre_date_max.date())
    end_time1 = time.time()
    training_time1 = (end_time1 - start_time1) / 60
    print("Total time: {:.2f}分钟".format(training_time1))

