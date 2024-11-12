import warnings
import tensorflow as tf
from matplotlib import rcParams
import pandas as pd

warnings.filterwarnings("ignore")

rcParams['font.family'] = 'SimHei'
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

global max_values1, max_values2, min_values1, min_values2, loaded_scaler_y, in_dim, out_dim, window_size_water, window_size_oil, group_id, data_num, sample_num, min_date, max_date, shift_num, out_num, in_num, \
    path, water_wellid, water_layerid, oil_wellid, water_features, water_layer_features, oil_features, label_features

path = './processed_data/'
water_features = ['prod_date', 'inj_daily', 'production_state_1']
oil_features = ['prod_date', 'capacity', 'production_state_2', 'oil_press', 'bottomhole_flow_press',
                'prod_duration', 'water_cut']
label_features = ['prod_date', 'sum_dailyoil']
group_id = 1

window_size_water = 90
window_size_oil = 10
shift_num = 30 # One month of water injection is recommended

# apso hyperparameters
maxiter_v = 30
swarmsize_v = 20
patience_v = 5
w_max= 0.9
w_min = 0.4
c1 = 2.0
c2 = 2.0

# Get well group relationships
water_wellid = []
oil_wellid = []
info_table = pd.read_excel(f'{path}/group_info.xlsx')  # Table fields include 'group_id', 'well_id', 'is_injection'
relate_data = info_table[info_table['group_id'] == group_id]
water_data = relate_data[relate_data['is_injection'] == 1]['well_id']
oil_data = relate_data[relate_data['is_injection'] == 0]['well_id']
water_wellid = list(set(water_data))
water_wellid.sort()
oil_wellid = list(set(oil_data))
oil_wellid.sort()

out_num = len(oil_wellid)
in_num = len(water_wellid)
in_dim = in_num * 2
out_dim = out_num * 6

# Optimal range of daily water injection
list_lb = []
list_ub = []