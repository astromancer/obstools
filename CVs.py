import re
import urllib.request
from html.parser import HTMLParser

import itertools as itt
from collections import OrderedDict
from misc import first_true_idx

class MyHTMLParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        HTMLParser.__init__(self, *args, **kwargs)
        self.data = ''
        self.links = []

    def handle_starttag(self, tag, attrs):
        #print("Start tag:", tag)
        #ipshell()
        for attr in attrs:
            ipshell()
            if attr[0] == 'href':
                if attr[1]: 
                    self.links.append( attr[1] )
            #print("     attr:", attr)

    #def handle_endtag(self, tag):
    #   print("End tag  :", tag)

    def handle_data(self, data):
        #if data not in ('\n',):
        self.data += data



master = 'http://observ.pereplet.ru/MASTER_OT.html'
response = urllib.request.urlopen( master )
html = response.read()
htmllines = html.split( b'\n' )

isnumbered = lambda s: s[0].isdigit()
lines = map( bytes.decode, htmllines )
lines = list(filter( None, lines ))
ix0 = first_true_idx( lines, isnumbered )
ix1 = -first_true_idx( reversed(lines), isnumbered )
data = lines[ix0:ix1]

#link_pattern = '<a {1,2}href="(?P<link>.*?) {0,2}".*?>(?P<text>.*?)</[aA]>'
#unlinker = re.compile( link_pattern )

patterns = (('nr',      '[\s\dART]{3}'),
            ('name',    'MASTER {1,2}J?[\d+-.]+'), 
            ('date',    '20\d{2} [\w.]{3,4} {1,3}[\d.]+'),
            ('type',    '[\w\|?/:]{1,5}'),
            ('mag',     '[\d.]{4,5}'),
            ('remark',  '[\w/]{0,6}'),
            ('ref',     '[ATel GCN PZP CBET IAUC #\d?.,]+'),
            ('site',    'SA|Amu|Tun|Arg|Dom|Kis|Net|Ura'),
            ('pioneer', '[\w+?]*'),
            ('comment', '.*') )
patterns = OrderedDict( patterns )
m = itt.starmap( '(?P<{}>{})'.format, patterns.items() )
master_pattern = '\s{0,5}'.join(m)
parser = re.compile( master_pattern )

success, links = [], []
fail, flinks = [], []
ixf = []
for i,l in enumerate(data):
    try:
        html_parser = MyHTMLParser(strict=False)
        html_parser.feed(l)
        
        mo = parser.match( html_parser.data )
        success.append( mo.groups() )
        links.append( html_parser.links )
    
    except Exception as err:
        fail.append( l )     
        ixf.append( i )
        flinks.append( html_parser.links )

print( '%s lines successfully parsed' % len(success) )
print( '%s lines failed' % len(fail) )
