# Visibility tracks for celestial bodies


```python

from obstools.plan.visibilities import VisPlot


vis = VisPlot(date='2020-07-27',  # default is today's date
              site='Cape Town',   # defualt is 'SAAO'
              targets=['CTCV J1928-5001',  # coordinates will be resolved automatically
                       'IGR J19552+0044',
                       '1RXS J231603.9-052713'])
vis.add_targets({'NEOWISE': '09 57 49 +46 28 56',
                 'NOI-102975': '16:42:06.1 -01:18:55.4',
                 'Swift J1839-0453': '18h 39m 20.0s -4 53 53.1'})
vis.connect()  # connect interactions (highlight on hover, remove by legend click etc.)
```

![Example Visibility Tracks](/obstools/plan/data/example.png "Example Visibility Tracks")
