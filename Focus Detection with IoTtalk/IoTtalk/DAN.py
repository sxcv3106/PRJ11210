import requests, time, random, threading
from iottalk import csmapi

profile = {
	'd_name': None,
	'dm_name': 'MorSensor',
	'u_name': 'yb',
	'is_sim': False,
	'df_lst': [],
}

def get_mac_addr():
	from uuid import getnode
	mac = getnode()
	mac = ''.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))
	return mac

timestamp={}
MAC=get_mac_addr()
thx=None
def register_device(addr):
    global MAC, profile, timestamp, thx

    if csmapi.ENDPOINT == None: detect_local_ec()

    if addr != None: MAC = addr

    if profile['d_name'] == None: profile['d_name']= str(int(random.uniform(1, 100)))+'.'+ profile['dm_name']

    for i in profile['df_list']: timestamp[i] = ''

    print('IoTtalk Server = {}'.format(csmapi.ENDPOINT))
    if csmapi.register(MAC,profile):
        print ('\nThis device has successfully registered.')
        print ('Device name = ' + profile['d_name'] + '\n')
 
        return True
    else:
        print ('Registration failed.')
        return False

def device_registration_with_retry(IP=None, addr=None):
    if IP != None:
        csmapi.ENDPOINT = 'https://' + IP
    success = False
    while not success:
        try:
            register_device(addr)
            success = True
        except Exception as e:
            print ('Attach failed: '),
            print (e)
        time.sleep(1)

def pull(FEATURE_NAME):
    global timestamp

    # if state == 'RESUME': data = csmapi.pull(MAC,FEATURE_NAME)
    # else: data = []
    data = csmapi.pull(MAC,FEATURE_NAME)

    if data != []:
        if timestamp[FEATURE_NAME] == data[0][0]:
            return None
        timestamp[FEATURE_NAME] = data[0][0]
        if data[0][1] != []:
            return data[0][1]
        else: return None
    else:
        return None

def deregister():
    return csmapi.deregister(MAC)
