# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:51:15 2023

@author: Soo.Y
"""


def change_raw_to_dict(raw_text):
    text = raw_text.split(',')
    status_dict = {}
    for item in text:
        area, status = item.split(":")
        status_dict[area.strip()] = status.strip()
    return status_dict
