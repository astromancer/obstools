import re
import warnings
from collections import UserList

import numpy as np
from astropy.coordinates import SkyCoord
#from recipes.iter import grouper

class Jparser(UserList):
    pattern = '([+-]*\d{1,2})(\d{2})(\d{2}\.?\d{0,3})'
    single_parser = re.compile(pattern)
    #full_parser = re.compile(pattern * 2)
    parser = re.compile(pattern * 2)

    def __init__(self, name):
        self.raw = name
        # extract the coordinate data
        match = self.parser.search(name)
        if match is None:
            raise ValueError('No coordinate match found!')

        self.data = match.group(1,2,3), match.group(4,5,6)

        #self.match_map = map( self.parser.findall, coolist )
        #self.group_map = map( lambda mo: mo.groups(), self.match_map )

    #def get_data():
        #match = self.parser.search(name)
        #data = match.group(1,2,3), match.group(4,5,6)
        #return data

    def to_string(self, sep=':'):
        coo_str_map = map(sep.join, self.data)
        return ' '.join(coo_str_map)

    def skycoord(self):
        return SkyCoord(self.to_string(), unit=('h', 'deg'))

    @classmethod
    def many2gen(cls, names):
        """Generator that yields tuples of hms, dms separated coordinates"""
        for name in names:
            data = cls.single_parser.findall(name)
            if len(data) > 2:
                #NOTE: some object designations like 'CRTS SSS100805 J194428-420209'
                #matches 3 times here.  We simply use the last two available
                yield data[:2]
            elif len(data) == 2:
                yield data
            else:
                warnings.warn('Unable to parse coordinates for %s' % name)
                yield ((None,)*3,)*2

    @classmethod
    def many2array(cls, names):
        """parse names into Nx2x3 array of hms, dms coords"""
        cooList = list(cls.many2gen(names))
        return np.array(cooList).astype(float)

    @classmethod
    def many2deg(cls, names):
        coords = cls.many2array(names)
        scaler = ((1 / 15, 60, 60 ** 2), (1, 60, 60 ** 2))
        return np.sum(coords / scaler, -1)


    @classmethod
    def many2str_gen(cls, names, sep=':'):
        for coo in cls.many2gen(names, sep=sep):
            if coo is None:
                yield None
            ra, dec = coo
            yield sep.join(ra), sep.join(dec)


    @classmethod
    def many2str(cls, names, sep=':'):
        return list(cls.many2str_gen(names, sep=sep))


    @classmethod
    def many2skycoord(cls, names):
        coos = map(' '.join, cls.many2str_gen(names))
        return [SkyCoord(coo, unit=('h', 'deg')) if coo else None for coo in coos]

    #def to_XEphem(self, **kw ):
        #return list( itt.starmap( to_XEphem, zip(self.raw, self.to_names()) ))



if __name__ == '__main__':
    # a few test cases:
    names = ['CRTS SSS100805 J194428-420209',
             '1RXS J174320.1-042953',
             'MASTER J095958.98-190100.6']
    for name in names:
        print(Jparser(name).to_string())

