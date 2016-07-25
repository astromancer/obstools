import re
import urllib.request
from html.parser import HTMLParser

import itertools as itt
from collections import OrderedDict
from recipes.iter import first_true_idx

from obstools.jparser import Jparser


class MyHTMLParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        HTMLParser.__init__(self, *args, **kwargs)
        self.data = ''
        self.links = []

    def handle_starttag(self, tag, attrs):
        #print("Start tag:", tag)
        #ipshell()
        for attr in attrs:
            #ipshell()
            if attr[0] == 'href':
                if attr[1]: 
                    self.links.append( attr[1] )
            #print("     attr:", attr)

    #def handle_endtag(self, tag):
    #   print("End tag  :", tag)

    def handle_data(self, data):
        #if data not in ('\n',):
        self.data += data


#Jparser

master = 'http://observ.pereplet.ru/MASTER_OT.html'
response = urllib.request.urlopen( master )
html = response.read()
htmllines = html.split( b'\n' )

isnumbered = lambda s: s[0].isdigit()
lines = map(bytes.decode, htmllines)    #map to str
lines = list(filter(None, lines))
ix0 = first_true_idx(lines, isnumbered)
ix1 = -first_true_idx(reversed(lines), isnumbered)
data = lines[ix0:ix1]

#meta = lines[ix1:]
#pions = first_true_idx(meta, lambda l: l.startswith('Pioneer'))
#pione = first_true_idx(meta, lambda l: 'Types of optical transients' in l)
#pioneers = re.findall('([A-Z]{2}) -',  ' '.join(meta[pions+1:pione]))

patterns = (('nr',      'ART|[ \d-]{4}'),               #ART? WTF!
            ('name',    '.+?'), 
            ('date',    '20\d\d [\w.]{3,4} {1,3}[\d.]+'),
            ('type',    '[\w\|?/:]{1,5}'),
            ('mag',     '[\d.]{0,5}'),
            ('remark',  '[\w/]{0,6}'),
            ('ref',     '.+'), #'[ATel GCN PZP CBET IAUC #\d?.,]+'),    #ATel|GCN|PZP|CBET|IAUC[ #\d?.,]+
            ('site',    'SA|Amu|Tun|Arg|Dom|Kis|Net|Ura|IAC|Iac'),
            ('pioneer', '[A-Z?]*'),
            ('comment', '.*') )
patterns = OrderedDict(patterns)
m = list(itt.starmap( '(?P<{}>{})'.format, patterns.items() ))
master_pattern = '\s{0,5}'.join(m[:])
parser = re.compile( master_pattern )


success, links = [], []
image_urls = []
fail, flinks = [], []
ixf = []
txt = []
for i,l in enumerate(data):
    try:
        html_parser = MyHTMLParser() #strict=False
        html_parser.feed(l)
        
        txt.append(html_parser.data)
        
        mo = parser.match(html_parser.data)
        vals = list(map(str.strip, mo.groups()))
        name = mo.groupdict()['name']
        
        success.append(vals )
        
        iurl = html_parser.links[0].strip() if len(html_parser.links) else None
        image_urls.append(iurl)
        
        links.append(html_parser.links)
    
    except Exception as err:
        fail.append( l )     
        ixf.append( i )
        flinks.append( html_parser.links )

print( '%s lines successfully parsed' % len(success) )
print( '%s lines failed' % len(fail) )


