import os
import urllib.request
from datetime import datetime, timedelta
import io
from PIL import Image


def get_suth_weather_png(path):
    """Retrieve and save png image of Sutherland environmental monitoring page"""
    addy = 'http://suthweather.saao.ac.za/image.png'
    response = urllib.request.urlopen(addy)
    data = response.read()
    stream = io.BytesIO(data)

    img = Image.open(stream)
    t = datetime.now()
    if 0 < t.hour < 12:  # morning hours --> backdate image to start day of observing run
        t -= timedelta(1)

    datestr = str(t.date()).replace('-', '')
    filename = os.path.join(path, 'env' + datestr + '.png')
    print(filename)
    img.save(filename)
