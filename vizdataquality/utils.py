# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 04:43:43 2023

   Copyright 2023 Roy Ruddle

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   
"""

import os
import datetime

import chardet
import logging


##############################################################################
def init_logging(logfile_name, overwrite_output_file=False):
    """Initialise a logfile.
    
    Logfiles can be useful when you are processing large datafiles or debugging.

    Parameters
    ----------
    logfile_name : str
        The full pathname of the logfile.
    overwrite_output_file : boolean, optional
        True (start a new logfile) or False (append to the file if it exists, and start a new file if it does not exist). The default is False.

    Returns
    -------
    log : Logger
        A Logger object.
    handlers : list
        The logfile's handlers.

    """
    error = False
    function_name = 'init_logging()'
    
    if overwrite_output_file or not os.path.isfile(logfile_name):
        # Add header
        try:
            with open(logfile_name, "w") as f:
                #f = open(logfile_name,'w')#"a")
                f.write("DATE\tTIME\tPATHNAME\tFILENAME\tLOGGER\tLINE\tLEVEL\tMESSAGE\r\n")
                #f.close() 
        except Exception as e:
            print(function_name + ' ' + type(e).__name__ + '. Unexpected exception: ' + getattr(e, 'message', str(e)))
            error = True

    if not error:   
        #logging configuration & Initialisation
            
        # Multi logging handlers    
        # https://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log
        log = logging.getLogger('vizdataquality')
        log.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s\t%(pathname)s\t%(module)s\t%(name)s\t%(lineno)s\t%(levelname)s\t%(message)s', datefmt='%d/%m/%Y\t%H:%M:%S')
        
        # create file handler which logs even debug messages
        fh = logging.FileHandler(filename=logfile_name)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
        
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        formatter2 = logging.Formatter('%(levelname)s\t%(message)s')
        ch.setFormatter(formatter2)
        log.addHandler(ch)
        
        handlers = [fh, ch]
    else:
        log = None
        handlers = None
        
    return log, handlers


##############################################################################
def end_logging(log, handlers):
    """End logging.

    Parameters
    ----------
    log : Logger
        The logger.
    handlers : TYPE
        The logfile handlers.

    Returns
    -------
    None.

    """
    # Remove handlers so logging messages is not output multiple times
    
    for handler in handlers:
        log.removeHandler(handler)


##############################################################################
def detect_file_encoding(input_filename, read_in_chunks=True, confidence_level=0.9):
    """Detect and return the encoding of a text file

    Parameters
    ----------
    input_filename : str
        The absolute path of the file.
    read_in_chunks : boolean, optional
        True (read the whole file; more accurate) or False (read the file until the confidence level is satisfied). Default is False. The default is True.
    confidence_level : float, optional
        Confidence threshold (only used if read_in_chunks = True). The default is 0.9.

    Returns
    -------
    result : dict
        A dictionary containing the 'encoding' and a 'confidence' level, or None (file could not be found/opened).

    """
    result = None
    function_name = "detect_file_encoding()"
    logger = logging.getLogger('QCprofiling')
    logger.debug("%s,input_filename,%s,read_in_chunks,%s,confidence_level,%s" %(function_name, input_filename, str(read_in_chunks), str(confidence_level)))
    st = datetime.datetime.now()
    
    try:
        with open(input_filename, "rb") as fin:

            if read_in_chunks:

                while True:

                    try:
                        chunk_size = 1024
                        rawdata = fin.read(chunk_size)
                        
                        if len(rawdata) > 0:
                            result = chardet.detect(rawdata)
                            
                            if result['confidence'] >= confidence_level:
                                break
                            else:
                                result = None

                        else:
                            break

                    except (SystemExit, KeyboardInterrupt):
                        logger.warning("%s KeyboardInterrupt" %(function_name))
                        raise
                    except Exception as e:
                        #logger.error('%s,Exception,%s' %(function_name,getattr(e, 'message', repr(e))))
                        logger.exception('%s %s. Unexpected exception: %s' %(function_name, type(e).__name__, getattr(e, 'message', str(e))))
                        raise

            else:

                try:
                    rawdata = fin.read()
                    
                    if len(rawdata) > 0:
                        result = chardet.detect(rawdata)

                except (SystemExit, KeyboardInterrupt):
                    logger.warning("%s KeyboardInterrupt" %(function_name))
                    raise
                except Exception as e:
                    logger.exception('%s %s. Unexpected exception: %s' %(function_name, type(e).__name__, getattr(e, 'message', str(e))))
                    raise

    except FileNotFoundError as e:
        logger.warning('%s Cannot open file: %s' %(function_name,input_filename))
    except Exception as e:
        logger.exception('%s %s. Unexpected exception: %s' %(function_name, type(e).__name__, getattr(e, 'message', str(e))))
        raise
    else:
        et = datetime.datetime.now()
        logger.info("%s,input_filename,%s,time (s),%s" %(function_name, input_filename, str((et-st).total_seconds())))
        
    return result
