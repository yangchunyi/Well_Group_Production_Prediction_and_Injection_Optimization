import pandas as pd
from matplotlib import rcParams
import matplotlib
import warnings
import numpy as np
import tensorflow as tf

global in_dim, out_dim, out_num, in_num, window_size_water, window_size_oil, group_id, sample_num, window_sizes
# Configuration
matplotlib.use('TkAgg')  # Select an appropriate backend, e.g., 'TkAgg' or 'Qt5Agg'
warnings.filterwarnings("ignore")
rcParams['font.family'] = 'SimHei'
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Random seed
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

# Hyperparameter settings
Epochs = 400
Batch_Size = 64
Learning_rate = 0.001
window_size_water = 90
window_sizes = [10, 20, 30]
test_ratio = 0.3
path = './processed_data/'
water_features = ['prod_date', 'inj_daily', 'production_state_1']
oil_features = ['prod_date', 'capacity', 'production_state_2', 'oil_press', 'bottomhole_flow_press',
                'prod_duration', 'water_cut']
label_features = ['prod_date', 'sum_dailyoil']
group_id = 1

# Get well group relationships
water_wellid = []
oil_wellid = []
info_table = pd.read_excel(f'{path}/group_info.xlsx')# Table fields include 'group_id', 'well_id', 'is_injection'
relate_data = info_table[info_table['group_id'] == group_id]
water_data = relate_data[relate_data['is_injection'] == 1]['well_id']
oil_data = relate_data[relate_data['is_injection'] == 0]['well_id']
water_wellid = list(set(water_data))
water_wellid.sort()
oil_wellid = list(set(oil_data))
oil_wellid.sort()

out_num = len(oil_wellid)
in_num = len(water_wellid)
list_water = []
list_oil = []
in_dim = in_num * 2
out_dim = out_num * 6