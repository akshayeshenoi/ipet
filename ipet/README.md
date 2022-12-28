# iPET

iPET has two components:  
1) **ipet-pert-train**: This directory contains the source code to train the generators that will be used by the victim.
3) **ipet-pert-add**: Once the generators are trained (i.e., model files saved), the user can call `predict.py` to produce perturbation vector files that instruct the gateway to add appropriate dummy packets. We also provide a sample script that adds these perturbations on undefended traffic.

## Pre-requisite
As described in the paper, the generator training process requires a discriminator to be trained first, which is the M_Sequence fingerprinting model. Therefore, a prerequisite is to train the M_Sequence (see `IoT_Device_Fingerprinting` in the root directory).

Follow the steps described [here](../IoT_Device_Fingerprinting/README.md) to train the discriminator.

## Training Generators
Once the discriminator is trained, the model files will be saved in `IoT_Device_Fingerprinting/Models/` where we will access it from.

Before training iPET, the following parameters can be configured in `constants.py` (present in the `ipet-pert-train/` directory):
- `max_packets_per_omega` : Maximum number of dummy packets allowed to be added in a discrete time-slot 
- `max_payload_per_omega` : Maximum additional payload bytes to be added in a discrete time-slot
- `training_stages` : Number of stages you want to train iPET for

Note that the discrete time-slot `omega` and total window time `total_time` must be configured for the discriminator M_Sequence.

Finally, the iPET generators can be trained using the following script: 
```sh
$ cd ipet/ipet-pert-train
$ python iPet_Training.py 
```

The generated models will be saved in the `ipet-pert-train/Models` directory.

### Producing Perturbations
The trained models will be saved on the disk for each a) device, b) generator version. The `predict.py` script loads the saved models for the specific combination of parameters requested by the user.  

Note: Please refer to Section 4 of our paper for a detailed explanation of how iPET perturbations are produced.

The script takes the following positional arguments
- `num_samples`: Total number of perturbations vectors required. A vector specifies the number of dummy packets to be added for each time slot. The total number of time slots per vector (`total_time`), and the duration of each time slot (`omega`) is configured [here](../IoT_Device_Fingerprinting/README.md#m_sequence). Therefore, under the [default](../IoT_Device_Fingerprinting/constants.py) settings, `num_samples = 10` would correspond to `10 x 4 = 40` seconds of perturbations, with each time slot being `0.1` seconds long.
- `dev_id`: The id of the device. In our example, we use human friendly IDs that correspond to the IDs used in the [training](../IoT_Device_Fingerprinting/constants.py) phase.
- `generation`: The generator stage G = 0,1,2.

For instance, to produce 10 perturbation vectors (concatenated), by generator at stage 2, for device `smrtthings`:
```sh
$ python predict.py 10 smrtthings 2
```

The output will be saved as a numpy array which can be used by the gateway to add perturbations. Each row corresponds to timeslot omega and states the quantities that must be added as cover traffic. Specifically, they are: 
`# outgoing packets | total size of outgoing packets | # incoming packets | total size of incoming packets`

### Adding Perturbations
To test the perturbations on real traffic, we provide a simple simulator.  

For each device, the script does the following:
1. It pre-generates perturbations, if they don't exist (using the same `predict.py` script above) and loads them.
2. It loads device-specific traffic traces from `../IoT_Device_Fingerprinting/data/split-trace/`.
3. It increments a timer and adds the perturbations (in the form of dummy packets) to the device traffic trace, as instructed by the vectors.
4. The output traffic trace is saved in numpy format.

The output contains metadata for each packet in the following format:
- The direction of the packet (1 for outgoing, -1 for incoming)
- Flag indicating whether it is a dummy packet (0 for legitimate, 1 for dummy)
- Size of the packet (in bytes)
- Time when it was sent (in seconds)

Packets are produced in both incoming and outgoing directions.

Parameters such as omega, number of days, number of samples and generator stage (generation) can be configured directly in the file.

To run the script, simply execute:
```sh
$ python ipet_pert_add.py
```

The traces are saved in the `output/` directory.
