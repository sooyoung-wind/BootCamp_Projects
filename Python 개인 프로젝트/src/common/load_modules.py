# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:20:17 2023

@author: Soo.Y
"""


def load_all_modules():
    try:
        import googlemaps
    except ImportError:
        import pip
        pip.main(['install', 'googlemaps'])
    finally:
        try:
            import googlemaps
            return True
        except ImportError:
            return False
