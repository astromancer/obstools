import socket
import logging

from recipes.misc import is_interactive
is_interactive = is_interactive()

def setup_logging():
    # ===============================================================================
    # Decide how to log based on where we're running
    if socket.gethostname().startswith('mensa'):
        plot_diagnostics = False
        print_progress = False
        log_progress = True
    else:
        plot_diagnostics = True  # True
        print_progress = True
        log_progress = False

    if is_interactive:  # turn off logging when running interactively (debug)
        from recipes.interactive import exit_register

        log_progress = print_progress = False
        monitor_mem = False
        monitor_cpu = False
        monitor_qs = False
    else:
        from atexit import register as exit_register
        from recipes.io.tracewarn import warning_traceback_on

        # check_mem = True            #prevent excecution if not enough memory available
        monitor_mem = True
        monitor_cpu = True  # True
        monitor_qs = True  # False#

        # setup warnings to print full traceback
        warning_traceback_on()
        logging.captureWarnings(True)

    # print section timing report at the end
    exit_register(chrono.report)

    # ===============================================================================
    # Logging setup # NOTE: atm this has to be done *before* importing the modelling lib

    # create directory for logging / monitoring data to be saved
    logpath = fitspath.with_suffix('.log')
    if not logpath.exists():
        logpath.mkdir()

    if plot_diagnostics:
        figpath = fitspath.with_suffix('.figs')
        if not figpath.exists():
            figpath.mkdir()

    # create logger with name 'phot'
    lvl = logging.DEBUG
    logbase = 'phot'
    logger = logging.getLogger(logbase)
    logger.setLevel(lvl)

    # create file handler which logs event debug messages
    logfile = str(logpath / 'phot.log')
    fh = logging.FileHandler(logfile, mode='w')
    # fh.setLevel(lvl)

    # create console handler with a higher log level
    # NOTE: this will log to both file and console
    logerr = str(logpath / 'phot.log.err')
    ch = logging.FileHandler(logerr, mode='w')
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    # create formatter and add it to the handlers
    fmt = '{asctime:<23} - {name:<32} - {process:5d}|{processName:<17} - {levelname:<10} - {message}'
    space = (23, 32, 5, 17, 10)
    indent = sum(space) * ' '
    formatter = logging.Formatter(fmt, style='{')
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    # logger.addHandler(ch)

    #
    # ModelDb.basename = logbase

    # ===============================================================================
    if monitor_mem or monitor_cpu or monitor_qs:
        mon_mem_file = logpath / 'phot.mem.dat'
        mon_cpu_file = logpath / 'phot.cpu.dat'
        mon_q_file = logpath / 'phot.q.dat'

    # Start process monitors if needed
    if monitor_cpu:  # TODO: monitor load???
        from recipes.parallel.utils import monCPU

        # start cpu performance monitor
        mon_cpu_alive = mp.Event()
        mon_cpu_alive.set()  # it's alive!!!!
        monproc_cpu = mp.Process(name='cpu monitor',
                                 target=monCPU,
                                 args=(mon_cpu_file, 0.1, mon_cpu_alive),
                                 daemon=True  # will not block main from exiting
                                 )
        monproc_cpu.start()

        @exit_register
        def stop_cpu_monitor():
            # stop cpu performance monitor
            mon_cpu_alive.clear()
            # monproc_cpu.join()
            # del monproc_cpu

            # atexit.register(stop_cpu_monitor)

    if monitor_mem:
        from recipes.parallel.utils import monMEM

        # start memory monitor
        mon_mem_alive = mp.Event()
        mon_mem_alive.set()  # it's alive!!!!
        monproc_mem = mp.Process(name='memory monitor',
                                 target=monMEM,
                                 args=(mon_mem_file, 0.5, mon_mem_alive),
                                 daemon=True  # will not block main from exiting
                                 )
        monproc_mem.start()

        @exit_register
        def stop_mem_monitor():
            # stop memory monitor
            mon_mem_alive.clear()
            # monproc_mem.join()
            # del monproc_mem


            # atexit.register(stop_mem_monitor)

    return print_progress, log_progress

