import itertools as itt
import numpy as np

from collections import OrderedDict


from magic.iter import interleave, consume, grouper
from magic.list import lmap

from IPython import embed

#****************************************************************************************************
class RKField(np.ndarray):

    def __new__(cls, data, uncertainty, empty, **special_flags):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(data).view(cls)
        # add the new attribute to the created instance
        obj.uncertainty = uncertainty
        obj.empty = empty
        for key, val in special_flags.items():
            setattr( obj, key, val )
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # Note that it is here, rather than in the __new__ method,
        # that we set the default values for attributes, because this
        # method sees all creation of default objects - with the
        # RKField.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.uncertainty        =       getattr(obj, 'uncertainty', None)
        self.empty              =       getattr(obj, 'empty', None)
        
        
#****************************************************************************************************
class RKCat( object ):          #TODO: use astropy.table
    def __init__(self, filename):
        self.Fields, self.Data = Fields, Data = self.read_data(filename)
        stat = []
        
        table = Table( Data.T, names=Fields )
        
        #embed()
        
        
        #for i, (field, data) in enumerate(zip(Fields, Data)):
            #special_flags = {}
            #cleaned, u_flag, empty = self.unflag( data )

            ##Period flag
            #if field=='P_orb':
                #SH_flag = [val.endswith('*') for val in cleaned]                                  #in case of object type SU or SH:  if followed by *, the orbital period has been estimated from the known superhump period using the empirical relation given by Stolz & Schoembs (1984, A&A 132, 187).
                #special_flags['SH_flag'] = SH_flag
                #cleaned = [s.strip('* ') for s in cleaned]
                

            #status, converted = self.convert(cleaned, empty)
            #stat.append(status)
            
            #data = RKField( converted, u_flag, empty, **special_flags )
            #setattr( self, field, data )
            #Data[i] = data
            
    
    def __getitem__(self, key):
        if isinstance( key, str ):
            l = self.name == key
            if np.any(l):
                return OrderedDict( zip(self.Fields, self.Data.T[l][0]) )
            else:
                raise KeyError( 'Object %s not found in catalogue...' % key )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def read_data(self, filename):
        count = 0
        #fields = [['name', 'RA', 'type1','type3', 'mag1', 'mag3', 'T1', 'P_orb', 'P3', 'EB', 'Spectr2', 'M1_M2', 'Incl', 'M1', 'M2'],
                    #['Alt_name', 'DEC', 'type2','type4', 'mag2', 'mag4', 'T2', 'P2', 'P4', 'SB', 'Spectr1', 'e_M1_M2', 'e_Incl', 'e_M1', 'e_M2']]
        
        fields = ['name', 'alias', 'flag1', 'flag2', 'ra', 'dec', 
                  'type1', 'type2', 'type3', 'type4', 'mag1', 'mag2', 'mag3', 'mag4',
                  'T1', 'T2', 'P0', 'P2', 'P3', 'P4', 'EB', 'SB', 'spectr2', 'spectr1',
                  'q', 'q_E', 'Incl',  'Incl_E', 'M1', 'M1_E', 'M2', 'M2_E'] 
        
        data = []
        with open(filename,'r') as fp:
            
            datalines = itt.filterfalse( lambda s: s.startswith(('\n','-')), fp )
            header = [next(datalines), next(datalines)]#consume(datalines, 2)       #header
            
            splitter = lambda line: map( str.strip, line.strip('\n|\r').split('|') )
            for i, lpair in enumerate( grouper(datalines, 2) ):
                data.append( interleave( *map( splitter, lpair ) ) )
                #data.append( dat )

        #Interleaving data and fields (every 2 lines refer to a single catalogue object)
        #data = [interleave(*s) for s in zip(*data)]
        #fields = interleave(fields)

        print('Data for %i objects successfully read.\n\n' %(i+1))

        return np.array(fields), np.array(data).T

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def unflag(self, dat):
        '''Generates boolean arrays for: data flagged for uncertainty; empty data values
        Returns cleaned data and flag arrays'''
        u_flag = np.array( [val.endswith((':','?')) for val in dat] )
        empty = np.array( [val == '' for val in dat] )
        if any(u_flag):
            #print 'u_flag:', field
            #eval(field).u_flag = u_flag
            cleaned = [s.strip(':? ') for s in dat]
            #dat = cleaned
        else:
            cleaned = dat

        return cleaned, u_flag, empty

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def convert(self, data, empty):
        '''Data type convertion'''
        missing_value = None
        data = np.array(data)
        
        #ipshell()
        #print( data, type(data) )
        try:
            data[empty] = missing_value
            converted = data.astype(float)
            status = True
        except ValueError:
            converted = data
            status = False

        return status, converted
    
    
def read_data(filename):
    count = 0
    #fields = [['name', 'RA', 'type1','type3', 'mag1', 'mag3', 'T1', 'P_orb', 'P3', 'EB', 'Spectr2', 'M1_M2', 'Incl', 'M1', 'M2'],
                #['Alt_name', 'DEC', 'type2','type4', 'mag2', 'mag4', 'T2', 'P2', 'P4', 'SB', 'Spectr1', 'e_M1_M2', 'e_Incl', 'e_M1', 'e_M2']]
    
    fields = ['name', 'alias', 'flag1', 'flag2', 'ra', 'dec', 
                'type1', 'type2', 'type3', 'type4', 'mag1', 'mag2', 'mag3', 'mag4',
                'T1', 'T2', 'P0', 'P2', 'P3', 'P4', 'EB', 'SB', 'spectr2', 'spectr1',
                'q', 'q_E', 'Incl',  'Incl_E', 'M1', 'M1_E', 'M2', 'M2_E'] 
    
    data = []
    with open(filename,'r') as fp:
        
        datalines = itt.filterfalse( lambda s: s.startswith(('\n','-')), fp )
        header = [next(datalines), next(datalines)]#consume(datalines, 2)       #header
        
        splitter = lambda line: map( str.strip, line.strip('\n|\r').split('|') )
        for i, lpair in enumerate( grouper(datalines, 2) ):
            data.append( interleave( *map( splitter, lpair ) ) )
            #data.append( dat )

    #Interleaving data and fields (every 2 lines refer to a single catalogue object)
    #data = [interleave(*s) for s in zip(*data)]
    #fields = interleave(fields)

    print('Data for %i objects successfully read.\n\n' %(i+1))
    #embed()
    #return Table( np.array(data), names=fields )
    return np.array(fields), np.array(data)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def unflag(dat):
    '''Generates boolean arrays for: data flagged for uncertainty; empty data values
    Returns cleaned data and flag arrays'''
    u_flag = np.array( [val.endswith((':','?')) for val in dat] )
    empty = np.array( [val == '' for val in dat] )
    if any(u_flag):
        #print 'u_flag:', field
        #eval(field).u_flag = u_flag
        cleaned = [s.strip(':? ') for s in dat]
        #dat = cleaned
    else:
        cleaned = dat

    return cleaned, u_flag, empty
        
#****************************************************************************************************
import re
from astropy.coordinates import SkyCoord
from collections import UserList

class Jparser(UserList):
    pattern = '([+-]*\d{2})(\d{2})(\d{2}\.\d{1,3})'
    parser = re.compile( pattern )

    def __init__(self, coolist):
        self.raw = coolist
        self.data = lmap( self.parser.findall, coolist )
        #self.match_map = map( self.parser.findall, coolist )
        #self.group_map = map( lambda mo: mo.groups(), self.match_map )

    def to_strings(self, sep=':'):
        str_map = lambda radec: sep.join( radec )
        coo_str_map = lambda coo: lmap( str_map, coo )
        return lmap( coo_str_map, self.data )

    def to_string(self, sep=':', joiner=' '):
        return lmap(' '.join, self.to_strings(sep) )

    def to_SkyCoords(self):
        return SkyCoord( self.to_string(), unit=('h', 'deg') )

    def to_XEphem(self, **kw ):
        return list( itt.starmap( to_XEphem, zip(self.raw, self.to_strings()) ))



def to_XEphem( *args, **kw ):
    
    if len(args)==3:
        kw['name'], kw['ra'], kw['dec'] = args
    elif len(args)==2:
        kw['name'] = args[0]
        kw['ra'], kw['dec'] = args[1]

    kw.setdefault( 'mag', 15 )
    kw.setdefault( 'epoch', 2000 )
    
    return '{name},f|V,{ra},{dec},{mag},{epoch}'.format( **kw )

if __name__ == '__main__':
    
    from astropy.table import Table
    from astropy.table.column import Column

    from pySHOC.timing import Time
    from astropy.time import TimeDelta
    from pySHOC.airmass import altitude
    
    #RKCat()
    #fields, data = read_data( '/media/Oceanus/UCT/Project/RKcat7.21_main.txt' )
    fields, data = read_data( '/media/Oceanus/UCT/Observing/RKCat7.23.main.txt' )
    #table = read_data( '/media/Oceanus/UCT/Project/RKcat7.21_main.txt' )
    uncertain = np.vectorize( lambda s: s.endswith((':','?')) )(data)
    cleaned = np.vectorize( lambda s: s.strip(':?') )(data)
    empty = np.vectorize( lambda s: s=='' )(data)
    
    table = Table( cleaned, names=fields )
    
    #convert RA/DEC columns This is a massive HACK!!
    ra, dec = table['ra'], table['dec']
    decmap = map(lambda dec: dec.rsplit(' ', 1)[0], dec)
    coords = list(map( ' '.join, zip(ra, decmap) ))
    #coords = Column( name='coords', data=SkyCoord( coords, unit=('h', 'deg') ) )
    coords = SkyCoord( coords, unit=('h', 'deg') )      #TODO: Try use this as column??
    ra_col = Column( coords.ra, 'ra', float )
    dec_col = Column( coords.dec, 'dec', float )
    i = table.colnames.index('ra')
    table.remove_columns( ('ra', 'dec') )
    table.add_columns( [ra_col, dec_col], [i,i] )
    
    #Type filter
    types = [c for c in table.colnames if 'type' in c]
    mtypes = ('AM', 'AS', 'LA', 'IP', 'LI')             #magnetic systems 'IP', 'LI'
    ltype = np.array( [any(t in mtypes for t in tt) for tt in table[types]] )
    
    #Hour angle filter
    #lra = (9 > coords.ra.hour) & (coords.ra.hour < 19)
    
    tq = table[ltype] #&lra
    #tq.sort('ra')
    
    raise SystemExit
    
    #Magnitude filter
    #mags = [c for c in table.colnames if 'mag' in c]
    mag1 = np.vectorize( lambda s: float(s.strip('>UBVRIKpgr') or 100) )(tq['mag1'])
    lmag = mag1 <= 18
    tqq = tq[lmag]
    
    #l = [ltype&lra][lmag]
    
    #Altitude filter
    t0 = Time('2015-12-03 00:00:00', scale='utc')
    interval = 300 #seconds
    td = TimeDelta(interval, format='sec')
    
    days = 3
    N = days*24*60*60 / interval
    t = t0 + td*np.arange(N)
    
    dawn, dusk = 8, 18
    lnight = (dusk < t.hours) | (t.hours < dawn)
    t = t[lnight]
    
    iers_a = t.get_updated_iers_table( cache=True )        #update the IERS table and set the leap-second offset
    delta, status = t.get_delta_ut1_utc( iers_a, return_status=True )
    t.delta_ut1_utc = delta
    

    #lat, lon = 18.590549, 98.486546    #TNO
    #28°45'38.3" N       17°52'53.9" W   +2332m
    lat, lon = 28.7606389, 17.88163888888889     #WHT
    lmst       = t.sidereal_time('mean', longitude=lon)
    
    ra = np.radians( tqq['ra'].data[None].T )
    dec = np.radians( tqq['dec'].data[None].T )
    altmax = altitude( ra,
                        dec,
                        lmst.radian,
                        np.radians(lat) ).max(1)
    lalt = np.degrees(altmax) >= 45
    
    #Orbital period filter
    pc = tqq['P0']
    pc[pc==''] = '-1'
    Ph = pc.astype(float) * 24
    lP = (Ph < 5) & (Ph > 0)
    
    tqqq = tqq[lalt&lP]
    tqqq.sort('ra')
    
    trep = tqqq.copy()
    trep.remove_columns( ('flag1', 'flag2', 'T1', 'T2', 'type4', 'P2', 'P3', 'P4', 'spectr1') )
    
    #
    coo_str = SkyCoord( ra=trep['ra'], dec=trep['dec'], unit='deg' ).to_string( 'hmsdms', sep=':') 
    ra_str, dec_str = zip(*map( str.split, coo_str ))
    ra_col = Column( ra_str, 'ra' )
    dec_col = Column( dec_str, 'dec' )
    
    trep.remove_columns( ('ra', 'dec') )
    trep.add_columns( [ra_col, dec_col], [2,2] )
    
    
    i = trep.colnames.index('P0')
    Pm = trep['P0'].astype(float) * 24 * 60
    col = Column( Pm, 'P Orb (min)', float )
    trep.remove_column( 'P0' )
    trep.add_column( col, i )
    
    trep.show_in_browser()
    
    