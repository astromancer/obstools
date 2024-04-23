"""
Detect 
"""

import numpy as np


def detect(run):

    is_cal = np.array(run.calls('pointing_zenith'))
    
    sections = {}
    for i, obs in enumerate(run):
        if obs.pointing_zenith():
            continue # calibration stack
        
        ha = obs.t.ha.hour
        l = (0 <= ha) & (ha <= 0.5)
        if l.any():
            sections[i] = l

    return sections

# def detect(stack):
    
