import numpy as np
import pandas as pd
from multiprocessing import Pool
import subprocess
from pathlib import Path
import random
import sys

OMEGA = 0.1 # second
op_path = "./output"
perts_path = "./perts"

num_samples = int(2e4)
num_days = 1
generation = 1

# prepare time ticks
# should be whenever there's a frame, or every 100ms
# ignore silence time to reduce output size
# reutrns [[timestamp, packetlen, direction]] packetlen == -1 for c.t.
def create_ticks(packet_timings, packet_lens, packet_dirs):
    packet_timings = np.array(packet_timings)
    packet_lens = np.array(packet_lens)
    packet_dirs = np.array(packet_dirs)

    # helper fn
    def create_subtrace(ts_buffer, silence_threshold):
        # silence period, create sub trace
        sub_trace = np.array(ts_buffer)
        # insert equally spaced ticks
        sub_trace_flat = sub_trace[:,[0]].reshape(-1)

        # repeat two times; one for outgoing and one for incoming (OMEGA/2 for 0.5)
        tick_trace_og = np.arange(sub_trace_flat[0] + OMEGA/2, sub_trace_flat[len(sub_trace_flat) - 1] + silence_threshold, OMEGA)
        tick_trace_og = [[tick_val, -1, 1] for tick_val in tick_trace_og] # -1 to indicate it's not a real packet, 1 to indicate it is outgoing

        tick_trace_ig = np.arange(sub_trace_flat[0], sub_trace_flat[len(sub_trace_flat) - 1] + silence_threshold, OMEGA)
        tick_trace_ig = [[tick_val, -1, -1] for tick_val in tick_trace_ig] # -1 to indicate it's not a real packet, -1 to indicate it is incoming

        # merge
        sub_trace = np.concatenate((sub_trace, tick_trace_og, tick_trace_ig,))
        # sort along the time axis
        sub_trace = sub_trace[sub_trace[:,0].argsort()]
        return sub_trace

    silence_threshold = 5 # 5 seconds

    # variables
    ts_buffer = []
    final_trace = []
    for i in range(len(packet_timings) - 1):
        ts_buffer += [[packet_timings[i], packet_lens[i], packet_dirs[i]]] # save packet_len and dir as well
        # if next ts is outside the silence_threshold, we enter silence period
        if (packet_timings[i+1] - packet_timings[i]) > silence_threshold:
            sub_trace = create_subtrace(ts_buffer, silence_threshold)

            # add to final_trace
            final_trace += [sub_trace]

            # reset
            ts_buffer = []
            continue

    # add last subtrace, if needed
    if len(ts_buffer) > 0:
        sub_trace = create_subtrace(ts_buffer, silence_threshold)
        final_trace += [sub_trace]

    final_trace_ts = np.concatenate(final_trace)
    return final_trace_ts

def ipet_trace(perts_arr, ticks):
    # adds the specified perturbations every OMEGA
    # -1 indicates that the tick is not a legitimate packet
    # send legitimate packets without waiting 

    # output [dir, ct, packetlen, time]
    output_arr = []
    pert_iter = -1

    # pert variables
    num_packets = -1
    total_payload = -1

    perts_total = len(perts_arr)

    def send(packet_to_send, tick_ts, direction, ct=False):
        nonlocal output_arr
        output_arr += [[direction, ct, packet_to_send, tick_ts]]

    for tick in ticks:
        direction = tick[2]
        user_packet = tick[1] != -1

        if user_packet:
            # send without waiting
            user_packet_len = int(tick[1])
            send(user_packet_len, tick[0], direction, ct=False)

        else:
            # send all cover traffic in this tick
            pert_iter += 1
            this_pert = perts_arr[pert_iter % perts_total]
            if direction == 1: # outgoing CONFIRM WITH KANAV
                num_packets = this_pert[2]
                total_payload = this_pert[3]

            else: # incoming
                num_packets = this_pert[0]
                total_payload = this_pert[1]

            if num_packets == 0:
                num_packets = 1

            pkt_size = total_payload / num_packets
            for _ in range(int(num_packets)):
                send(pkt_size, tick[0], direction, ct=True)

    return np.array(output_arr)


def driver(df):
    print(df[1])
    trace_df = df[0]
    device_name = df[1].split('.')[0].split('_')[2]
    pert_npy_fp =  '_'.join([device_name, str(generation)]) + '.npy'

    if not Path(perts_path + "/" + pert_npy_fp).is_file():
        # generate perturbation
        op = subprocess.run(['python3', 'predict.py', str(num_samples), device_name, str(generation)], stderr=subprocess.DEVNULL)
        # ,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        op = subprocess.run(['mv', pert_npy_fp, perts_path + "/" + pert_npy_fp])

    perts_arr = np.load(perts_path + "/" + pert_npy_fp)

    ticks = create_ticks(trace_df['frame.time_relative_normalised'],
                            trace_df['transport.len'],
                            trace_df['dir'])

    op_arr = ipet_trace(perts_arr, ticks)
    
    np.save(op_path + '/' + df[1], op_arr)


original_trace_bp = '../../IoT_Device_Fingerprinting/data/split-trace/'

devices = pd.read_csv(original_trace_bp + '../devices.csv', delimiter='\t')

ignore_list = ['smgglxtb', 'andrphb', 'laptop', 'macbook', 'andrph', 'iphone', 'maciphone']
devices_list = list(set(devices['Hostname']) - set(ignore_list))

experiment_list = []

# number of days
for day in range(num_days):
    for device in devices_list:
        exp_case = 'day_' + str(day) + '_' + device + '.csv'
        experiment_list += [exp_case]

dfs_list = []

for experiment in experiment_list:
    try:
        og_trace_df = pd.read_csv(original_trace_bp + experiment, delimiter=',')
        # separate incoming and outgoing traffic
        # 1 for outgoing, -1 for incoming
        og_trace_df['dir'] = og_trace_df['ip.src'].str.contains("^192.*$").apply(lambda x: 1 if x else -1)

        # start time_relative from 1st packet
        og_trace_df['frame.time_relative_normalised'] = og_trace_df['frame.time_relative'] - og_trace_df['frame.time_relative'][0]

        dfs_list += [[og_trace_df, experiment]]

    except FileNotFoundError:
        # print("FileNotFoundError:", experiment)
        pass

Path(op_path).mkdir(exist_ok=True)
Path(perts_path).mkdir(exist_ok=True)

random.shuffle(dfs_list)

POOL_SIZE = 5

pool = Pool(POOL_SIZE)
pool.map(driver, dfs_list)
