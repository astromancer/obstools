

# std libs
import textwrap
from io import BytesIO

# third-party libs
import aplpy
import pyfits
import numpy as np
import astropy.coordinates as astcoo



def get_coords( obj_name ):
    ''' Attempts a SIMBAD Sesame query with the given object name. '''
    try: 
        print( '\nQuerying SIMBAD database for {}...'.format(repr(obj_name)) )
        coo = astcoo.name_resolve.get_icrs_coordinates( obj_name )
        ra = coo.ra.to_string( unit='h', precision=2, sep=' ', pad=1 )
        dec = coo.dec.to_string( precision=2, sep=' ', alwayssign=1, pad=1 )

        print( 'The following ICRS J2000.0 coordinates were retrieved:\nRA = {}, DEC = {}\n'.format(ra, dec) )
        return coo, ra, dec

    except Exception as err:     #astcoo.name_resolve.NameResolveError
        print( 'ERROR in retrieving coordinates...' )
        print( err )
        return None, None, None


def get_dss(imserver, ra, dec, epoch=2000, size=(10,10)):
    '''
    Grab a image from STScI server and pull it into pyfits.
    '''
    h, w = size
    url = textwrap.dedent('''            http://archive.stsci.edu/cgi-bin/dss_search?
            v=%s&
            r=%f&d=%f&
            e=J%f&
            h=%f&w=%f&
            f=fits&
            c=none''' % (imserver, ra, dec, epoch, h, w) ).replace( '\n', '' )
    fitsData = BytesIO()
    data = urllib.request.urlopen(url).read()
    fitsData.write(data)
    fitsData.seek(0)
    return pyfits.open(fitsData, ignore_missing_end=True)


def draw_line(plot, theta, length, ra, dec, **kw):
    '''draw a line centered at ra,dec of a given length at a given angle'''
    #set default keywords
    kw.setdefault('color','b')
    kw.setdefault('linewidth',1)
    kw.setdefault('alpha', 0.7)

    theta = theta*np.pi/180.0
    length = length/2.0
    dx = np.sin(theta)*length/(np.cos(dec*np.pi/180.0)*60.0)
    dy = np.cos(theta)*length/60.0
    coords = np.array([[ra+dx, ra-dx], [dec+dy, dec-dy]])

    plot.show_lines([coords], **kw)
    return plot



servname = {'none'                  : '',
            'poss2ukstu_red'        : "POSS2/UKSTU Red",
            'poss2ukstu_blue'       : "POSS2/UKSTU Blue",
            'poss2ukstu_ir'         : "POSS2/UKSTU IR",
            'poss1_blue'            : "POSS1 Blue",
            'poss1_red'             : "POSS1 Red",        }



def make_finder(obj_name, size=(10,10)):
    coo, _, _ = get_coords( obj_name )
    ra, dec = coo.ra.deg, coo.dec.deg

    #get image
    allowed_servers = ('poss2ukstu_blue', 'poss1_blue', 
                        'poss2ukstu_red', 'poss1_red', 
                        'poss2ukstu_ir',
                        'all'                    )
    for imserver in allowed_servers:
        try:
            print( 'Retrieving FITS image from DSS server: %s' %imserver )
            hdu = get_dss(imserver, ra, dec, size=size)
            break
        except Exception as err:
            print( 'DSS image retrieval failed with:\n%s\n' %err )
    
    plot = aplpy.FITSFigure(hdu)
    plot.show_grayscale()
    plot.set_theme( 'publication' )
    
    plot.add_label( 0.5, 1.03,
                    obj_name,
                    relative=True, style='italic', weight='bold', size='large',
                    layer='text' )
    plot.add_label( -0.05, -0.05, 
                    "%s" % servname[imserver],
                    relative=True, style='italic', weight='bold',
                    layer='labels' )

    plot.add_grid()
    plot.grid.set_alpha(0.2)
    plot.grid.set_color('b')

    # add cardinal direction labels
    plot.add_label(ra, dec + 4.8/60.0,
                  "N",
                  style='italic',
                  weight='bold',
                  size='large',
                  color=(0,0.5,1) )     #layer='labels'
    plot.add_label(ra + 4.8/(np.abs(np.cos(dec*np.pi/180.0))*60),
                  dec,
                  "E",
                  style='italic',
                  weight='bold',
                  size='large',
                  horizontalalignment='right',
                  color=(0,0.5,1) )     #layer='labels'
    #Draw N-S crosshair
    w, h = size
    for angle, length in zip([0,90], size):
        print( angle, length )
        plot = draw_line(plot, angle, w, ra, dec, color='g', linewidth=0.5, alpha=1.0) #layer='crosshair'
    
    return plot
