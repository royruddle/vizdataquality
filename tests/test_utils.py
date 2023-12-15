import pytest
import tempfile
import os
from vizdataquality.utils import init_logging, end_logging

def test_create_new_log_file(tmp_path):
    log_file = tmp_path / "test_log.log"
    log, _ = init_logging(str(log_file), True)
    assert log_file.exists()
    with open(log_file, 'r') as file:
        header = file.readline()
        expected_header = ("DATE\tTIME\tPATHNAME\tFILENAME\tLOGGER\t"
                           "LINE\tLEVEL\tMESSAGE")
        assert expected_header in header

def test_end_logging():
    with tempfile.NamedTemporaryFile(delete=False) as temp_log:
        log_file = temp_log.name

    log, handlers = init_logging(log_file, True)
    initial_handler_count = len(log.handlers)

    end_logging(log, handlers)
    # Close all handlers and clear them
    for handler in log.handlers:
        handler.close()
    log.handlers.clear()

    if os.path.exists(log_file):
        os.remove(log_file)

