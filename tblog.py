# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 18:03:40 2021

@author: wli3535
"""

from tensorboard import default
from tensorboard import program


tracking_address = "C:/Users/jxr180022/Documents/Microgrid_Reconfiguration_RL/logger/R1_Microgrid_env_mlp_Normal_8"
#r1_34_bus_mlp_with_entropy_05_multi_env11_17_13

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None,'--logdir',tracking_address])
    tb.main()