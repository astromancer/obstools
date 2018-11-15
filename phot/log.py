import logging
import logging.config
import logging.handlers
import multiprocessing as mp


# import random

# DEBUG_LEVELV_NUM = 9
# logging.addLevelName(DEBUG_LEVELV_NUM, "DEBUGV")
# def debugv(self, message, *args, **kws):
#     # Yes, logger takes its '*args' as 'args'.
#     if self.isEnabledFor(DEBUG_LEVELV_NUM):
#         self._log(DEBUG_LEVELV_NUM, message, args, **kws)
# logging.Logger.debugv = debugv


class MyHandler(object):
    """
    A simple handler for logging events. It runs in the listener process and
    dispatches events to loggers based on the name in the received record,
    which then get dispatched, by the logging system, to the handlers
    configured for those loggers.
    """

    def handle(self, record):
        logger = logging.getLogger(record.name)
        # The process name is transformed just to show that it's the listener
        # doing the logging to files and console
        # record.processName = '%s (for %s)' % (mp.current_process().name, record.processName)
        logger.handle(record)


def listener_process(q, stop_event, config):
    """
    This could be done in the main process, but is just done in a separate
    process for illustrative purposes.

    This initialises logging according to the specified configuration,
    starts the listener and waits for the main process to signal completion
    via the event. The listener is then stopped, and the process exits.
    """
    logging.config.dictConfig(config)

    # print(logging.handlers)
    # root = logging.getLogger()
    # print('root handlers', root.handlers)

    listener = logging.handlers.QueueListener(q, MyHandler())
    logger = logging.getLogger()
    # print('setup handlers', logger.handlers)
    #
    logger.info('Starting listener')
    listener.start()

    # if os.name == 'posix':
    #     # On POSIX, the setup logger will have been configured in the
    #     # parent process, but should have been disabled following the
    #     # dictConfig call.
    #     # On Windows, since fork isn't used, the setup logger won't
    #     # exist in the child, so it would be created and the message
    #     # would appear - hence the "if posix" clause.
    #     logger = logging.getLogger('setup')
    #
    #     logger.critical('Should not appear, because of disabled logger ...')
    stop_event.wait()
    listener.stop()


def worker_init(config):  # counter,
    """
    This initialises logging for the worker according to the specified
    configuration,
    """
    # print('1')
    # logging.config.dictConfig(config)
    # print('2')
    logging.config.dictConfig(config)
    logger = logging.getLogger('setup')
    logger.info('Initializing: %s', mp.current_process().name)

    # counter.inc()
    #
    # with open('worker%d' % counter.get_value(), 'w') as fp:
    #     fp.write('hello world')

    # print('3')
    # try:
    # print(vars(logger))
    # logger.info('Initializing')
    # except Exception as err:
    #     import traceback
    #     traceback.print_exc()

    # print('4')

    # if os.name == 'posix':
    #     # On POSIX, the setup logger will have been configured in the
    #     # parent process, but should have been disabled following the
    #     # dictConfig call.
    #     # On Windows, since fork isn't used, the setup logger won't
    #     # exist in the child, so it would be created and the message
    #     # would appear - hence the "if posix" clause.
    #     logger = logging.getLogger('setup')
    #     logger.critical('Should not appear, because of disabled logger ...')


def config(logpath, q):
    # Logging setup

    # The main process gets a simple configuration which prints to the console.
    config_initial = {
        'version': 1,
        'formatters': {
            'detailed': {
                'class': 'logging.Formatter',
                'format': '%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s'
            }
        },
        'handlers': {
            'queue': {
                'class': 'logging.handlers.QueueHandler',
                'queue': q,
            },
            # console
        },
        'root': {
            'level': 'INFO',
            'handlers': ['queue']
        },
    }

    # The worker process configuration is just a QueueHandler attached to the
    # root logger, which allows all messages to be sent to the queue.
    # We disable existing loggers to disable the "setup" logger used in the
    # parent process. This is needed on POSIX because the logger will
    # be there in the child following a fork().
    config_worker = {
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'queue': {
                'class': 'logging.handlers.QueueHandler',
                'queue': q,
            },
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['queue']
        },
    }

    # The listener process configuration shows that the full flexibility of
    # logging configuration is available to dispatch events to handlers however
    # you want.
    # We disable existing loggers to disable the "setup" logger used in the
    # parent process. This is needed on POSIX because the logger will
    # be there in the child following a fork().
    fmt = '{asctime:<23} {name:<32} {process:5d}|{processName:<17} ' \
          '{levelname:<10} {message}'
    config_listener = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'class': 'logging.Formatter',
                'format': fmt,
                'style': '{',
            },
            'console': {
                'class': 'logging.Formatter',
                'format': '{asctime:<10} {processName:<17} {name:<32}  {'
                          'levelname:<10} {message}',
                # 'datefmt': '%H:%M:%S',
                'style': '{',
            },
            # 'barebones': {
            #     'class': 'logging.Formatter',
            # }
        },
        # 'filters': {
        #     'debugfilter': {
        #         '()': SingleLevelFilter,
        #         'passlevel': logging.DEBUG,
        #         'reject': True
        #     }
        # },
        'handlers': {
            'console': {  # log INFO (or higher) messages to console
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'console',
                # 'filters': ['debugfilter']

            },
            'file': {  # log *ALL* messages to *.log file
                'class': 'logging.FileHandler',
                'filename': str(logpath / 'main.log'),
                'mode': 'w',
                'formatter': 'detailed',
            },
            'errors': {  # log ERROR messages to *.errors.log file
                'class': 'logging.FileHandler',
                'filename': str(logpath / 'errors.log'),
                'mode': 'w',
                'level': 'ERROR',
                'formatter': 'detailed',
            },
            # 'progress': {  # log progress messages to console
            #     'class': 'logging.StreamHandler',
            #     'level': 'INFO',
            #     'formatter': 'barebones',
            #
            # },
            # 'progress.file': {  # log progress messages to console
            #     'class': 'logging.FileHandler',
            #     'level': 'INFO',
            #     'formatter': 'barebones',
            #
            # },
        },
        # 'loggers': {
        #     'progress': {
        #         'handlers': ['progress']
        #     }
        # },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'errors']
        },
    }

    return config_initial, config_listener, config_worker

    # logger = logging.getLogger('setup')
    # logger.info('About to create listener ...')
    # stop_event = mp.Event()
    # lp = mp.Process(target=listener_process, name='listener',
    #                 args=(q, stop_event, config_listener))
    # lp.start()
    # logger.info('Started listener')
    #
    # return lp, stop_event, config_worker
