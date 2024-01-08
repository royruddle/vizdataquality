# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from vizdataquality import report as vdqr

def test_report():
    report = vdqr.Report()
    assert isinstance(report.get_report_dict(), dict) == True
    
    key = report.add_heading('')
    assert isinstance(report.get_report_dict()[key], dict) == True
    
    key = report.add_acknowledgements('')
    assert isinstance(report.get_report_dict()[key], dict) == True
    
    key = report.step1()
    assert isinstance(report.get_report_dict()[key], dict) == True
    
    key = report.step2()
    assert isinstance(report.get_report_dict()[key], dict) == True
    
    key = report.step3()
    assert isinstance(report.get_report_dict()[key], dict) == True
    
    key = report.step4()
    assert isinstance(report.get_report_dict()[key], dict) == True
    
    key = report.step5()
    assert isinstance(report.get_report_dict()[key], dict) == True
    
    key = report.step6()
    assert isinstance(report.get_report_dict()[key], dict) == True
    
    key = report.dataset_size('', 0, 0)
    assert isinstance(report.get_report_dict()[key], dict) == True
    
    key = report.add_descriptive_stats(pd.DataFrame.from_dict({'A': [0]}))
    assert isinstance(report.get_report_dict()[key], dict) == True
    
    key = report.add_table(filename='')
    assert isinstance(report.get_report_dict()[key], dict) == True
    
    key = report.add_figure('')
    assert isinstance(report.get_report_dict()[key], dict) == True
    
    key = report.paragraph('')
    assert isinstance(report.get_report_dict()[key], dict) == True
