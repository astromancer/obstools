import logging

# create module level logger
logbase = 'phot'
logname = f'{logbase}.{__name__}'
logger = logging.getLogger(logname)   #__name__
logger.setLevel(logging.DEBUG)


# class PhotHelper:
#     """helper class for photometry interface"""
