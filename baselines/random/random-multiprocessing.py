import numpy as np
import pandas as pd
import random
from multiprocessing import Pool
from pathlib import Path

NUM_DAYS = 1 # days worth of csv data
NUM_RUNS = 1 # in case we want to run more than once
POOL_SIZE = 1 # parallel processes (to speed up)

T = 1 # window length
RANDOMIZE_INTERVAL = T * 5 # when should we select a new random R target
TICK = .1 # when to add bytes within a window
CT_BIAS = 0.5 # q from STP (0 means only mask when device is active, 1 means ILP (all the time))
# ^decides if cover traffic should be added

# parameters for the normal distribution
MU = 12.5e3
SIGMA = 4

# prepare time ticks - points in time something should happen.
# we basically process the pcap csv and insert ticks at times we may want to add cover traffic.
# there should be a tick whenever there's a legitimate packet, or every 100ms
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

        # repeat two times; one for outgoing and one for incoming
        tick_trace_og = np.arange(sub_trace_flat[0], sub_trace_flat[len(sub_trace_flat) - 1] + silence_threshold, TICK)
        tick_trace_og = [[tick_val, -1, 1] for tick_val in tick_trace_og] # -1 to indicate it's not a real packet, 1 to indicate it is outgoing

        tick_trace_ig = np.arange(sub_trace_flat[0], sub_trace_flat[len(sub_trace_flat) - 1] + silence_threshold, TICK)
        tick_trace_ig = [[tick_val, -1, -1] for tick_val in tick_trace_ig] # -1 to indicate it's not a real packet, -1 to indicate it is incoming

        # merge
        sub_trace = np.concatenate((sub_trace, tick_trace_og, tick_trace_ig,))
        # sort along the time axis
        sub_trace = sub_trace[sub_trace[:,0].argsort()]
        return sub_trace

    silence_threshold = 10*T

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

# returns [[direction, ct, packet_to_send, tick_ts]]
def Perturb(ticks):
    # output [dir, ct, packetlen, time]
    output_arr = []

    # variables
    ct_active = False # period when cover traffic is active
    buffered_user_activity = [] # buffer packets in case we run out of quota
    quota_t_left = -1 # initially. how much we can afford to spend over some T
    quota_tick = quota_t_left / (T/TICK) # initially. how much we can afford to spend over between ticks
    quota_tick /= 2 # because bidirectional
    ct_start = -1
    T_begin = ticks[0][0] # keep track of Ts

    _RANDOM_NUMBERS_LIST_SIZE = int(10e6) # to speed it up
    _random_numbers = np.abs(SIGMA * np.random.randn(_RANDOM_NUMBERS_LIST_SIZE) + MU)
    _random_iterator = 0

    def set_random_rate():
        nonlocal _random_iterator, _random_numbers, quota_t_left, quota_tick
        # simple uniform distribution
        quota_t_left = _random_numbers[_random_iterator % _RANDOM_NUMBERS_LIST_SIZE]
        quota_tick = quota_t_left / (T/TICK)
        quota_tick /= 2 
        _random_iterator += 1

    def should_start_ct():
        # simple Bernoulli distribution
        return random.random() < CT_BIAS

    def toggle_ct_active_on(tick_ts):
        nonlocal ct_active, ct_start
        ct_active = True
        ct_start = tick_ts

    def send(packet_to_send, tick_ts, direction, ct=False):
        nonlocal output_arr
        output_arr += [[direction, ct, packet_to_send, tick_ts]]

    # initialize
    set_random_rate()

    for tick in ticks:
        # check if at least one is true:
        #   1. user activity happens
        #   2. len(buffered_user_activity) != 0
        #   3. ct_active is True
        direction = tick[2]
        user_packet = tick[1] != -1
        if user_packet or len(buffered_user_activity) > 0 or ct_active:
            if not ct_active:
                # it's a user packet or a buffered packet
                # toggle on and initialize
                toggle_ct_active_on(tick[0])

            # first send any buffered packet
            buffered_user_activity_cpy = buffered_user_activity[:]
            for i in range(len(buffered_user_activity_cpy)):
                buff_packet = buffered_user_activity_cpy[i]
                if buff_packet[0] <= quota_t_left:
                    # can fit entirely
                    packet_to_send = buff_packet[0]
                    send(packet_to_send, tick[0], buff_packet[1], ct=False)
                    # update quota
                    quota_t_left -= packet_to_send

                elif quota_t_left > 0:
                    # send clipped packet
                    packet_to_send = quota_t_left
                    send(packet_to_send, tick[0], direction, ct=False)
                    # buffer remaining
                    buffered_user_activity += [[buff_packet[0] - packet_to_send, buff_packet[1]]]
                    # update quota
                    quota_t_left = 0

                else:
                    # out of quota
                    break

                # remove just sent packet
                buffered_user_activity.pop(0)

            # send user packet (if quota is available, else buffer)
            if user_packet:
                user_packet_len = int(tick[1])
                if user_packet_len <= quota_t_left:
                    # can fit entirely
                    packet_to_send = user_packet_len
                    send(packet_to_send, tick[0], direction, ct=False)
                    # update quota
                    quota_t_left -= packet_to_send

                elif quota_t_left > 0:
                    # send clipped packet
                    packet_to_send = quota_t_left
                    send(packet_to_send, tick[0], direction, ct=False)
                    # buffer remaining
                    buffered_user_activity += [[user_packet_len - packet_to_send, direction]]
                    # update quota
                    quota_t_left = 0

                else:
                    # buffer the whole thing
                    buffered_user_activity += [[user_packet_len, direction]]

            # send cover traffic (if quota available)
            elif quota_tick <= quota_t_left:
                packet_to_send = quota_tick
                send(packet_to_send, tick[0], direction, ct=True)
                # update quota, though there's no use at this point
                quota_t_left -= quota_tick

            elif quota_t_left > 0:
                packet_to_send = quota_t_left
                send(packet_to_send, tick[0], direction, ct=True)
                # update quota, though there's no use at this point
                quota_t_left -= quota_t_left

        if not ct_active:
            ct_active = should_start_ct()
            if ct_active:
                # if toggled on, initialize
                toggle_ct_active_on(tick[0])

        # should we stop cover_traffic
        if ct_active:
            if tick[0] - ct_start >= T:
                ct_active = False

        # randomize
        if tick[0] - T_begin >= RANDOMIZE_INTERVAL:
            T_begin = tick[0]
            set_random_rate()


    # there might be some traffic buffered up, just add more ticks
    last_tick = tick[0]
    # add a failsafe so it's not endless
    failsafe_max_iterations = int(1e7) # 1 millions iterations max
    failsafe_i = 0
    while True:
        ticks_extra = last_tick  + np.arange(len(buffered_user_activity)) * TICK
        set_random_rate()

        if failsafe_i > failsafe_max_iterations:
            break

        for tick in ticks_extra:
            failsafe_i += 1
            # just use last quota_t_left
            # note: we don't need to ensure the overshoot condition (like above loop)
            # since the tick instances here are purely separated by TICK distance.
            quota_tick_left = quota_t_left

            # first send any buffered packet
            buffered_user_activity_cpy = buffered_user_activity[:]
            for i in range(len(buffered_user_activity_cpy)):
                buff_packet = buffered_user_activity_cpy[i] 
                if buff_packet[0] <= quota_tick_left:
                    # can fit entirely
                    packet_to_send = buff_packet[0]
                    direction = buff_packet[1]
                    send(packet_to_send, tick, direction, ct=False)
                    # update quota
                    quota_tick_left -= packet_to_send

                elif quota_tick_left > 0:
                    # send clipped packet
                    packet_to_send = quota_tick_left
                    send(packet_to_send, tick, direction, ct=False)
                    # buffer remaining
                    buffered_user_activity += [[buff_packet[0] - packet_to_send, buff_packet[1]]]
                    # update quota
                    quota_tick_left = 0

                else:
                    # out of quota
                    break

                # remove just sent packet
                buffered_user_activity.pop(0)

        # are we done?
        if len(buffered_user_activity) == 0:
            break

        # still not done, update last tick and start over
        last_tick = tick


    return output_arr

def driver(experiment):
    print(experiment[1], experiment[2])
    trace_df = experiment[0]
    # ticks array
    # ticks[:,0] represents time, ticks[:,1] represents the packet_len at that time
    # if ticks[:,1] == -1, it means there's no user packet, but we can add cover traffic
    ticks = create_ticks(trace_df['frame.time_relative_normalised'],
                        trace_df['transport.len'],
                        trace_df['dir'])
    output_arr = Perturb(ticks)
    output_arr_np = np.array(output_arr)

    np.save('output_' + str(experiment[2]) + '/' + experiment[1] + '.npy', output_arr_np)

# load stuff
original_trace_bp = '../../IoT_Device_Fingerprinting/data/split-trace/'

devices = pd.read_csv(original_trace_bp + '../devices.csv', delimiter='\t')

ignore_list = ['smgglxtb', 'andrphb', 'laptop', 'macbook', 'andrph', 'iphone', 'maciphone']
devices_list = list(set(devices['Hostname']) - set(ignore_list))

experiment_list = []

for day in range(NUM_DAYS):
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


output_dfs_list = []

# create output folders
for i in range(NUM_RUNS):
    Path("./output_" + str(i)).mkdir(exist_ok=True)

# remove some if needed
# should look like what `df_exps` looks like below
done = []
if Path('done_list').is_file():
    with open('done_list') as fd:
        done = fd.read().split('\n')

erroring = []  # skip some if needed

ignore = done + erroring

exp_final = []

for df_pair in dfs_list:
    # add NUM_RUNS versions (save it in 4th index)
    df_exps = []
    for i in range(NUM_RUNS):
        if (df_pair[1] + '.npy.' + str(i) in ignore):
            print("Skipping ", df_pair[1], str(i))
            continue

        df_exps += [df_pair + [i]]

    exp_final += df_exps

random.shuffle(exp_final) # just so that the same process doesn't get all slow ones

pool = Pool(POOL_SIZE)
pool.map(driver, exp_final)

