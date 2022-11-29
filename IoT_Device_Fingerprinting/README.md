# IoT-Device-Fingerprinting
In Section 2.1, we describe two types of classifiers <img src="https://latex.codecogs.com/gif.latex?\mathcal{M}_\text{seq}" /> (M_Sequence) and <img src="https://latex.codecogs.com/gif.latex?\mathcal{M}_\text{agg}" /> (M_aggregate).

This directory contains the code that trains these models.

## Data required

### Data Collection
Packet capture (`pcap`) files captured from the vantage point of a home gateway (e.g. router) are required. For iPET, the setup must include devices communicating with one or more servers over the internet. The `pcap` may be split across days for convenience. 

Further, the `pcaps` must be converted into the `csv` format following a naming convention as `day[x].csv`, and must contain the following columns:
 - `frame.number`
 - `frame.time_relative`
 - `ip.src`
 - `ip.dst`
 - `ip.proto`
 - `tcp.len`
 - `tcp.stream`
 - `udp.length`
 - `eth.src`
 - `eth.dst`

Hint: The [tshark](https://www.wireshark.org/docs/man-pages/tshark.html) tool can help with the above operation.

A sample file is provided in the Releases tab.

More samples of the network traces can be found [here](https://drive.google.com/drive/folders/1gRkcrPupkYTWvYlgkkDKDmsP2FsJzG-g?usp=sharing).

Additionally, we also require a `devices.csv` file in the `data/` directory. A sample file (corresponding to the shared data) describes the format.

### Data Preparation
The day-wise `csv` traces must be split into day-wise _and_ device-wise traces. Execute the following script:
```sh
$ python split_traces.py
```

The models can now be trained.

## Training
### M_Sequence

**Configuration**  
The following parameters can be configured in `constants.py` for the M_sequence classifier:
- `total_time`: The total observation time for the time series, in seconds.
- `omega` : The duration of a discrete time-slot in the time series, in seconds.
- `train_sit`: To specify the model architecture to be used for training the fingerprinting classifier. Currently we only accept  `'LSTM', 'GRU', 'CNN' and 'MLP'` as arguements.
- `device_name` : Name list of the devices in the network. For e.g. `['device_A','device_B,'device_C']`

**Generating Fetaure Vectors**  
The raw data (described above) are converted to numpy feature vectors for the model to training on by running the script:
```sh
$ python Feature_Generation_Mseq.py 
```

The feature vectors generated are saved in the `data/Traces_Mseq` directory

#### Training M_Sequence
We train the sequential fingerprinting model using the following script: 
```sh
$ python Train_Mseq.py 
```
The trained model is saved in the `Models/M_seq_fingeprinting` directory

### M_Aggregate

**Configuration**  
To allow a user to customise their iPet instance, we expect them to specify the following variables in `constants.py`:
- `agg_mins`: The total time a single feature vector aggregates on , in minutes
- `device_name` : Name list of the devices in the network. For e.g. `['device_A','device_B,'device_C']`

**Generating Fetaure Vectors**  
The raw data is converted to numpy feature vectors for the model to training on by running the script:
```sh
$ python Feature_Generation_Magg.py 
```

The feature vectors generated are saved in the `data/Traces_Magg` directory

#### Training M_Aggregate
We train the sequential fingerprinting model using the following script: 
```sh
$ python Train_Magg.py 
```
The trained model is saved as `Models/M_agg_fingeprinting.sav` 

You can access the model using
```py
import pickle
model_magg = pickle.load(open('Models/M_agg_fingeprinting.sav', 'rb'))
```

These models are ready to be used as discriminators for [iPET](../ipet/README.md).
