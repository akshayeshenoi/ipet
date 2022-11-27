# iPET: Privacy Enhancing Traffic Perturbations for IoT Communications

iPET is a privacy enhancing traffic perturbation technique that counters ML-based fingerprinting attacks. iPET uses adversarial deep learning, specifically, Generative Adversarial Networks (GANs), to generate these perturbations. Unlike conventional GANs, a key idea of iPET is to deliberately introduce stochasticity in the model.

Our paper describes the workings of iPET in detail. [Link to paper]().

We make our source code available in this repository.

The code is organized as follows:
1. The `ipet` directory contains the core algorithm of our proposal (described in Section 4).
2. The `IoT_Device_Fingerprinting` directory contains the code for the attacker fingerprinting models (described in Section 2).
3. `baselines` contains code for traffic obfuscation techniques used in our evaluations (Section 5.1.2).

Each component contains relevant README files.

## Dependencies
We have tested our code on Ubuntu 20.04 and Python 3.8. Further dependencies can be installed from `environment.yml`.

```sh
$ conda env create --file environment.yml
$ conda activate ipet
```

## Contact
For any queries, please feel free to raise issues or contact the authors.