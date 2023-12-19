import pytest
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
    log_file = 'test_log.log'
    log, handlers = init_logging(log_file, True)
    initial_handler_count = len(log.handlers)

    end_logging(log, handlers)
    # Check if handlers added in this test are removed
    assert len(log.handlers) == initial_handler_count - len(handlers)

    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except PermissionError as e:
            print(f"Error removing log file: {e}")

