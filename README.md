# Reinforcment Learning (RL) for Microgrid Network Reconfiguration 
 This repository hosts the scripts to execute a reinforcement learning driven microgrid network reconfiguration scheme in both normal and contingency state.

## How to use
Users can reconfigure any given microgrid with minor modifications in the provided scripts and get insightful results by following the instructions below:
* "new_training.py" is the main script to run. It can be run via any python IDE or GUI on a machine which has OpenDSS already installed.
* The trained models are stored in "logger" directory under the master directory.
* The script tries to utilize all the existing cores of a cpu-based machine. The users need to set the correct number of cpu defined under the main function in the "new_training.py" script. For GPU-based machines, necessary modifications need to be made.
* "openDSSEnv34.py" is the script where the environment has been built. For each changing scenario, the reset function accommodates all the inputs like changing loading, line outages, etc.
* To check the desired output from a trained model, the test 'loading condition' and 'line outage' can be given through the ``test_initialize" function in the DSS_Initialize.py script.

### Dataset
The OpenDSS IEEE-34 bus network data was taken and modified to test the RL-based microgrid reconfiguration.

### Environment & Necessary Software
The framework has been built in python 3.8 and it uses OpenDSS in the backend. To implement the RL strategy to execute the reconfiguration task, the OpenAI Gym environment was utilized. Users are recommended to install those software to execute the scripts. Currently the scripts are only supported in Windows OS.

### Python library
```
pytorch              1.10.2
numpy                1.20.3
pandas               1.3.4
Stable Baselines3    1.3.0
OpenAI Gym           0.19.0
```


## Publications
**If you use these scripts in your research, please cite our publications**:

Rahman, J., Jacob, R. A., Paul, S., Chowdhury, S., & Zhang, J., Reinforcement Learning Enabled Microgrid Network Reconfiguration Under Disruptive Events, IEEE Kansas Power and Energy Conference (KPEC), Manhattan, Kansas, April 25 - 26, 2022.


**Collaborations are always welcome if more help is needed.**
## License
MIT License, Copyright (c) 2022 Jubeyer Rahman & Jie Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Contact

Jubeyer Rahman
jxr180022@utdallas.edu

Jie Zhang
jiezhang@utdallas.edu 

