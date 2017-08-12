#!/usr/bin/env python  
import math
import numpy as np
from matplotlib import pyplot as plt

freq = np.arange(0,1000,0.1)
chirp = [math.sin(2*math.pi*w) for w in freq ]

plt.plot(freq, chirp) 
