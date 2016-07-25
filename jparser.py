import re
from collections import UserList

from astropy.coordinates import SkyCoord
#from recipes.iter import grouper

class Jparser(UserList):
    pattern = '([+-]*\d{1,2})(\d{2})(\d{2}\.?\d{0,3})'
    single_parser = re.compile(pattern)
    #full_parser = re.compile(pattern * 2)
    parser = re.compile(pattern * 2)

    def __init__(self, string):
        self.raw = string
        #extract the coordinate data
        match = self.parser.search(string)
        if match is None:
            raise ValueError('No coordinate match found!')
        
        self.data = match.group(1,2,3), match.group(4,5,6)
        
        #self.match_map = map( self.parser.findall, coolist )
        #self.group_map = map( lambda mo: mo.groups(), self.match_map )    
    
    #def get_data():
        #match = self.parser.search(string)
        #data = match.group(1,2,3), match.group(4,5,6)
        #return data
    
    def to_string(self, sep=':'):
        coo_str_map = map(sep.join, self.data)
        return ' '.join(coo_str_map)
    
    def skycoord(self):
        return SkyCoord(self.to_string(), unit=('h', 'deg'))
    
    @classmethod
    def many2gen(cls, strings, sep=':'):
        datamap = map(cls.single_parser.findall, strings)
        for data in datamap:
            if len(data)>2:     
                #NOTE: some object designations like 'CRTS SSS100805 J194428-420209'
                #matches 3 times here.  We simply use the last two available
                yield data[:2]
            elif len(data)==2:
                yield data
            else:
                yield None
    
    @classmethod
    def many2str_gen(cls, strings, sep=':'):
        for coo in many2gen(cls, strings, sep=':'):
            if coo is None:
                yield None
            ra, dec = coo
            yield sep.join(ra), sep.join(dec)
    
    
    @classmethod
    def many2str(cls, strings, sep=':'):
        return list(many2str_gen(cls, strings, sep=':'))
    
    
    @classmethod
    def many2skycoord(cls, strings):
        coos = map(' '.join, cls.many2str_gen(strings))
        
        return [SkyCoord(coo, unit=('h', 'deg')) if coo else None for coo in coos]

    #def to_XEphem(self, **kw ):
        #return list( itt.starmap( to_XEphem, zip(self.raw, self.to_strings()) ))


        
if __name__ == '__main__':
    # a few test cases:
    names = ['CRTS SSS100805 J194428-420209', 
             '1RXS J174320.1-042953', 
             'MASTER J095958.98-190100.6']
    for name in names:
        print(Jparser(name).to_string())
   
    