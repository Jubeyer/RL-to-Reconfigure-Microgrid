# To run the RL_2_Reconfigure framework, the following steps need to be followed:
 i)"new_training.py" is the main script to run. It can be run via any python IDE or GUI on a machine which has OpenDSS already installed.
 ii) The trained models are stored in "logger" directory under the master directory.
iii) The script tries to utilize all the existing cores of a cpu based machine. The users need to set the correct number of cpu defined under the main function in the "new_training.py" script. For  GPU based machine necessary modifications need to be made.
iv) "openDSSEnv34.py" is the script where the environment has been built. For each changing scenarios, the reset function accommodates all the inputs like changing loading, line outages, etc.
v) To check the desired output from a trained model, the test 'loading condition' and 'line outage' can be given through the ``test_initialize" function in the DSS_Initialize.py script.
