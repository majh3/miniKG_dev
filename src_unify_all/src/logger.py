                                  

import logging
import sys

                                                                   

LOGGER = logging.getLogger('GlobalAppLogger')
LOGGER.setLevel(logging.DEBUG)
_LOGGING_SETUP_COMPLETE = False


def setup_logging(log_file_path, level=logging.INFO, hide_prefix=False):
    pass                                            

                                
                        
                                                      
       
    global _LOGGING_SETUP_COMPLETE
    if _LOGGING_SETUP_COMPLETE:
        return
    if LOGGER.hasHandlers():
        LOGGER.handlers.clear()

                              
    
               
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
                  
    simple_formatter = logging.Formatter('%(message)s')

                  
    if hide_prefix:
        formatter = simple_formatter
    else:
        formatter = detailed_formatter
        
                            
    
                
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)               
    LOGGER.addHandler(file_handler)

                          
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)               
                                                
    LOGGER.addHandler(stream_handler)

                                         
                                  
    LOGGER.info(f"Logger initialized. Writing logs to: {log_file_path}", stacklevel=2) 
    _LOGGING_SETUP_COMPLETE = True


                                                   

def _log_message(level_func, *args, **kwargs):
    if not args:
        return
    msg = " ".join(map(str, args))
    kwargs['stacklevel'] = kwargs.get('stacklevel', 2) 
    level_func(msg, **kwargs)

def debug(*args, **kwargs):
    _log_message(LOGGER.debug, *args, **kwargs)

def info(*args, **kwargs):
    _log_message(LOGGER.info, *args, **kwargs)

def warning(*args, **kwargs):
    _log_message(LOGGER.warning, *args, **kwargs)

def error(*args, **kwargs):
    _log_message(LOGGER.error, *args, **kwargs)

def critical(*args, **kwargs):
    _log_message(LOGGER.critical, *args, **kwargs)