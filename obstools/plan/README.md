# Visibility tracks for celestial bodies

## Basic Example

```python
from obstools.plan import SkyTracks


viz = SkyTracks(date='2020-07-27',  # default is today's date
                site='Cape Town',   # defualt is 'SAAO'
                targets=['CTCV J1928-5001',  # coordinates will be resolved automatically
                         'IGR J19552+0044',
                         '1RXS J231603.9-052713'])
viz.add_targets({'NEOWISE': '09 57 49 +46 28 56',
                 'NOI-102975': '16:42:06.1 -01:18:55.4',
                 'Swift J1839-0453': '18h 39m 20.0s -4 53 53.1'})
```

![Example Visibility Tracks](/obstools/plan/tests/images/test_readme_example_0.png "Example Visibility Tracks") 
<br /><br />

## Including telescope limits
Observing limits for SAAO 1.0m and 1.9m telescopes can be visualsed by 
initializing `SkyTracks` with `tel=1.` or `tel=1.9`.
```python
viz = SkyTracks(targets=['AR Sco', 'QS Tel', 'FO Aqr', 'FL Cet'],
                tel='1m')
```
![Visibility Tracks with Telescope Limits](/obstools/plan/tests/images/test_readme_example_1.png "Visibility Tracks with Telescope Limits")
<br /><br />

## Plot interaction
To connect plot interactions use:

```
viz.connect()
```

This will add a vertical line and clock indicating the current time to the plot.
Mouse interactions will also be activated.

Current interactions include
* highlight track when hovering mouse over legend entry
* remove track by clicking legend entry

To disable the interactive plot elements use:
```
viz.close()
```
