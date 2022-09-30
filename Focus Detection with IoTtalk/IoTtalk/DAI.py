import time, requests, random, json
from iottalk import DAN
from datetime import datetime
import collections
import keras
# import os
import numpy as np

Server_IP = 'iottalk2.ntcu.edu.tw'
Reg_addr = None

DAN.profile['dm_name'] = 'scc_sensor_data' #名稱
DAN.profile['df_list'] = ['scc_sensor_data'] #功能列表(包含IDF跟ODF)
DAN.profile['d_name'] = None

def get_mac_addr():
	from uuid import getnode
	mac = getnode()
	mac = ''.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))
	return mac

def register_to_iottalk():
	DAN.device_registration_with_retry(Server_IP, Reg_addr);

def dai(input):
    
    while True:
        try:
            for i in DAN.profile['df_list']:
                value = DAN.pull(i)
                if value != None:
                    value2 = str(value)
                    
                    temp = value2[1 : (len(value2)-1)]
                    s1,s2,s3 = temp.split(',')
                    
                    input.append(s1)
                    input.append(s2)
                    input.append(s3)
                    
                    return input
                    
        except Exception as e:
            print(e)
            DAN.device_registration_with_retry(Server_IP, Reg_addr)

        time.sleep(0.2)
