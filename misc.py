from urllib.request import urlretrieve
from pathlib import Path
from datetime import datetime

def get_suth_env(path):
    path = Path(path)
    ds = '{:d}{:02d}{:d}'.format(*datetime.today().utctimetuple()[:3])
    filename = 'env{}.png'.format(ds)
    url = 'http://suthweather.saao.ac.za/image.png'
    urlretrieve(url, str(path/filename))

#get_suth_env('/media/Oceanus/UCT/Observing/data/July_2016/log/')