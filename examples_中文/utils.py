"""
Utility functions for angle conversions used in the D1 examples.

This module provides:
  - deg_list_to_rad(xs): convert a list of angles in degrees to radians
  - rad_list_to_deg(xs): convert a list of angles in radians to degrees

Import these helpers in examples to keep the main scripts focused on
robot logic instead of unit conversions.
"""
import math

def deg_list_to_rad(xs):
    return [math.radians(v) for v in xs]

def rad_list_to_deg(xs):
    return [math.degrees(v) for v in xs]