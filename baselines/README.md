# Defense Baselines

In Section 5.1.2, we compare iPET with other traffic obfuscation techniques. The code for those techniques is present in this directory.

Each directory contains a script which includes the core traffic obfuscation algorithm and a driver function which simulates the obfuscation on real traffic. The `multiprocessing` library is used to parallelize perturbation.

Broadly, the scripts perform the following functions:
1. It loads device specific traffic traces from `../IoT_Device_Fingerprinting/data/split-trace/`
2. It maintains a timer, increments it and outputs both legitimate packets and/or cover traffic as instructed by the perturbation algorithm.
3. The output traffic is saved in numpy format.

### Data Required
Ensure that the data preparation step described in the `IoT_Device_Fingerprinting` [readme](../IoT_Device_Fingerprinting/README.md) is completed.

The scripts require day and device specific traces in the `../IoT_Device_Fingerprinting/data/split-trace` directory.

### Running
Each script can be run from the directory, e.g.:

```sh
$ cd constant
$ python constant-multiprocessing.py
```

If required, specific constants can be modified at the top of the file.

The output contains the direction of the packet, status as dummy packet, size of the packet and time when it was sent.

Note that packets are produced in both incoming and outgoing directions.

Also note that the output file may be significantly large depending on real traffic and/or perturbation parameters.
