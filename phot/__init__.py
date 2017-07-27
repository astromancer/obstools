import logging

# create module level logger
logbase = 'phot'
logname = '{}.{}'.format(logbase, __name__)
logger = logging.getLogger(logname)   #__name__
logger.setLevel(logging.DEBUG)
