import pandas as pd
import numpy as np
from multiprocessing import Pool
import os 
from sklearn.utils import shuffle
import constants
import helper

def get_magg_features(experiment_tuple, agg_mins = constants.agg_mins):
    
    day,device = experiment_tuple
    file_name = 'data/split-trace/day_'+str(day)+'_'+str(device)+'.csv'
    df = pd.read_csv(file_name)
    df['packet_polarity'] = df['ip.src'].apply(helper.check_polarity)
    df = df[['packet_polarity','tcp.len','udp.length','frame.time_relative']]

    final = pd.DataFrame(columns=['in_No_TCP','in_TCP_packet_length_avg','in_TCP_packet_length_min',
    'in_TCP_packet_length_max','in_tcp_flow_duration','out_No_TCP','out_TCP_packet_length_avg','out_TCP_packet_length_min',
    'out_TCP_packet_length_max','out_tcp_flow_duration','in_No_UDP','in_UDP_packet_length_avg','in_UDP_packet_length_min',
    'in_UDP_packet_length_max','in_UDP_flow_duration','out_No_UDP','out_UDP_packet_length_avg','out_UDP_packet_length_min',
    'out_UDP_packet_length_max','out_UDP_flow_duration','Device_ID'])

    final_length = 0
    for time in range(96):
        temp = df[(df['frame.time_relative']>=time*agg_mins*60) & (df['frame.time_relative']<=60*(time+1)*agg_mins)]
        if temp.shape[0]>0:
            in_temp_tcp = temp[(temp['packet_polarity']==-1) & (temp['tcp.len']>0)]
            if in_temp_tcp.shape[0]>0:
                in_tcp_flow_duration = in_temp_tcp['frame.time_relative'].iloc[-1] - in_temp_tcp['frame.time_relative'].iloc[0]
                in_no_tcp = in_temp_tcp.shape[0]
                in_avg_tcp_len = in_temp_tcp['tcp.len'].mean()
                in_tcp_min_length = in_temp_tcp['tcp.len'].min()
                in_tcp_max_length = in_temp_tcp['tcp.len'].max()
            else:
                in_tcp_flow_duration = 0
                in_no_tcp = 0
                in_avg_tcp_len = 0
                in_tcp_min_length = 0
                in_tcp_max_length = 0
                
            
            in_temp_udp = temp[(temp['packet_polarity']==-1) & (temp['udp.length']>0)]
            if in_temp_udp.shape[0]>0:
                in_udp_flow_duration = in_temp_udp['frame.time_relative'].iloc[-1] - in_temp_udp['frame.time_relative'].iloc[0]
                in_no_udp = in_temp_udp.shape[0]
                in_avg_udp_len = in_temp_udp['udp.length'].mean()
                in_udp_min_length = in_temp_udp['udp.length'].min()
                in_udp_max_length = in_temp_udp['udp.length'].max()
            else:
                in_udp_flow_duration = 0
                in_no_udp = 0
                in_avg_udp_len = 0
                in_udp_min_length = 0
                in_udp_max_length = 0
            
            
            out_temp_tcp = temp[(temp['packet_polarity']==1) & (temp['tcp.len']>0)]
            if out_temp_tcp.shape[0]>0:
                out_no_tcp = out_temp_tcp.shape[0]
                out_tcp_flow_duration = out_temp_tcp['frame.time_relative'].iloc[-1] - out_temp_tcp['frame.time_relative'].iloc[0]
                out_avg_tcp_len = out_temp_tcp['tcp.len'].mean()
                out_tcp_min_length = out_temp_tcp['tcp.len'].min()
                out_tcp_max_length = out_temp_tcp['tcp.len'].max()
            else:
                out_no_tcp = 0
                out_tcp_flow_duration = 0
                out_avg_tcp_len = 0
                out_tcp_min_length = 0
                out_tcp_max_length = 0

            out_temp_udp = temp[(temp['packet_polarity']==1) & (temp['udp.length']>0)]
            if out_temp_udp.shape[0]>0:
                out_no_udp = out_temp_udp.shape[0]
                out_udp_flow_duration = out_temp_udp['frame.time_relative'].iloc[-1] - out_temp_udp['frame.time_relative'].iloc[0]
                out_avg_udp_len = out_temp_udp['udp.length'].mean()
                out_udp_min_length = out_temp_udp['udp.length'].min()
                out_udp_max_length = out_temp_udp['udp.length'].max()
            else:
                out_no_udp = 0
                out_udp_flow_duration = 0
                out_avg_udp_len = 0
                out_udp_min_length = 0
                out_udp_max_length = 0

            new_entry = [in_no_tcp, in_avg_tcp_len,in_tcp_min_length,in_tcp_max_length, 
            in_tcp_flow_duration,out_no_tcp, out_avg_tcp_len,out_tcp_min_length,out_tcp_max_length,
            out_tcp_flow_duration,in_no_udp, in_avg_udp_len,in_udp_min_length,in_udp_max_length, 
            in_udp_flow_duration,out_no_udp, out_avg_udp_len,out_udp_min_length,out_udp_max_length,
            out_udp_flow_duration,int(constants.actual_id[device])]
            final.loc[final_length] = new_entry
            final_length+=1
    
    save_to = 'data/Traces_Magg/day'+ str(day)+'_'+str(device)+'_feat.csv'
    final.to_csv(save_to,index=False)

    return final

def collate_agg_traces():    

    experiments_all_local = []
    for day in range(constants.num_days):
        for device in constants.device_name:
            file_name = 'data/Traces_Magg/day'+ str(day)+'_'+str(device)+'_feat.csv'
            if os.path.isfile(file_name):
                experiments_all_local.append((day,device))

    X = pd.DataFrame()
    for (day,device) in experiments_all_local:
        temp = pd.read_csv('data/Traces_Magg/day'+ str(day)+'_'+str(device)+'_feat.csv')
        X = pd.concat([X,temp])
    
    X = shuffle(X)
    X.to_csv('data/Traces_Magg/M_agg_features.csv',index=False)
    
    for day in range(constants.num_days):
        for device in constants.device_name:
            if os.path.isfile('data/Traces_Magg/day'+ str(day)+'_'+str(device)+'_feat.csv'):
                os.remove('data/Traces_Magg/day'+ str(day)+'_'+str(device)+'_feat.csv')
    
    return None

def main():
    # check if we have all the split traces
    # if not, split them
    # if len(os.listdir('data/split-trace')) == 0:
    #     print("Splitting original traffic files......")
    #     helper.split_traces()

    experiments_all = []
    # for all days
    for day in range(constants.num_days):
        for device in constants.device_name:
            file_name = 'data/split-trace/day_'+str(day)+'_'+str(device)+'.csv'
            if os.path.isfile(file_name):
                experiments_all.append((day,device))


    print("Generating feature vectors......")

    pool = Pool(6)
    pool.map(get_magg_features, experiments_all)
    pool.close()
    pool.join()

    print("Collating feature vectors......")

    collate_agg_traces()

if __name__ == '__main__':    
    main()