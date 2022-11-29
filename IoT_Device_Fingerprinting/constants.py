num_days = 1
device_name = ['smrtthings','echo','netatwel','tpcloudcmra','smgsmrtcam','dropcam','instcama',
               'instcamb','wthbabymtr','belkinswch','tpsmrtplg','ihome','belkinmotn','nestsmkalrm',
               'netatwthst','wthsmrtscl','bldsgrmtr','wthaura','lifxblb','tribyspk','pxstrframe','hpprint',
               'smgglxtb','nstdropcm','andrphb','laptop','macbook','andrph','iphone','maciphone']

keys = device_name
values = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
actual_id = dict(zip(keys,values))
## M_Seq params
total_time = 4
omega = 0.1
train_sit = 'LSTM'  #'CNN'/'GRU'/'MLP'

## M_Agg params
agg_mins = 15