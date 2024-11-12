from setting import *

def make_sample(params):
    merged_water_list = []
    for i in range(len(water_wellid)):
        value_inj = params[i]
        data_in_all = pd.read_excel(
            f'{path}{group_id}/inject/{water_wellid[i]}.xlsx', usecols=water_features)
        data_in_all['prod_date'] = pd.to_datetime(data_in_all['prod_date'], format="%Y-%m-%d")
        data_in = data_in_all[['inj_daily', 'production_state_1']].tail(window_size_water - shift_num)

        data_in = data_in.reset_index(drop=True)

        new_row_data = {'inj_daily': value_inj, 'production_state_1': 0}
        new_row_data = pd.DataFrame(new_row_data, index=[0])
        repeated_rows = pd.concat([new_row_data] * shift_num, ignore_index=True)
        data_in.reset_index(inplace=True, drop=True)
        merged_water = pd.concat([data_in, repeated_rows], ignore_index=True)
        merged_water_list.append(merged_water)

    for i in range(len(merged_water_list)):
        merged_water_temp = merged_water_list[i]
        merged_water_temp.reset_index(inplace=True, drop=True)
        if i == 0:
            merged_water = merged_water_temp
        else:
            merged_water = pd.concat([merged_water, merged_water_temp], axis=1)

    for i in range(len(oil_wellid)):
        data_out_all = pd.read_excel(
            f'{path}{group_id}/output/{oil_wellid[i]}.xlsx', usecols=oil_features)
        data_out_all['prod_date'] = pd.to_datetime(data_out_all['prod_date'], format="%Y-%m-%d")

        data_out = data_out_all[['capacity', 'production_state_2', 'oil_press', 'bottomhole_flow_press',
                    'prod_duration', 'water_cut']].tail(1)
        data_out.reset_index(inplace=True, drop=True)

        if i == 0:
            merged_oil = data_out
        else:
            merged_oil = pd.concat([merged_oil, data_out], axis=1)

    new_rows = pd.concat([merged_oil] * window_size_oil, ignore_index=True)
    merged_oil = new_rows
    test_xdata1 = merged_water
    test_xdata2 = merged_oil
    return test_xdata1, test_xdata2