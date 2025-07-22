"""
Daily Net Radiation Integration (Verma et al., 2016)
====================================================

This module provides a function to integrate instantaneous net radiation to daily values using solar geometry parameters.
It is based on the methodology described in Verma et al. (2016) for global surface net radiation estimation from MODIS Terra data.

Key Features:
-------------
- Integrates instantaneous net radiation (Rn) to daily values using hour of day, latitude, and solar angles.
- Accepts Raster, numpy array, or float inputs for geospatial and scientific workflows.
- Handles calculation of daylight hours and sunrise time if not provided.

Reference:
----------
Verma, M., Fisher, J. B., Mallick, K., Ryu, Y., Kobayashi, H., Guillaume, A., Moore, G., Ramakrishnan, L., Hendrix, V. C., Wolf, S., Sikka, M., Kiely, G., Wohlfahrt, G., Gielen, B., Roupsard, O., Toscano, P., Arain, A., & Cescatti, A. (2016). Global surface net-radiation at 5 km from MODIS Terra. Remote Sensing, 8, 739. https://api.semanticscholar.org/CorpusID:1517647

Example Usage:
--------------
>>> from daily_Rn_integration_verma import daily_Rn_integration_verma
>>> Rn_daily = daily_Rn_integration_verma(Rn=400, hour_of_day=12, doy=180, lat=35)
"""

from typing import Union
import warnings
import numpy as np
from rasters import Raster
from sun_angles import daylight_from_SHA, sunrise_from_SHA, SHA_deg_from_DOY_lat

def daily_Rn_integration_verma(
        Rn: Union[Raster, np.ndarray, float],
        hour_of_day: Union[Raster, np.ndarray, float],
        DOY: Union[Raster, np.ndarray, float] = None,
        lat: Union[Raster, np.ndarray, float] = None,
        sunrise_hour: Union[Raster, np.ndarray, float] = None,
        daylight_hours: Union[Raster, np.ndarray, float] = None
        ) -> Union[Raster, np.ndarray, float]:
    """
    Calculate daily net radiation using solar parameters.

    This represents the average rate of energy transfer from sunrise to sunset
    in watts per square meter. To get the total energy transferred, multiply
    by the number of seconds in the daylight period (daylight_hours * 3600).

    Reference:
        Verma, M., Fisher, J. B., Mallick, K., Ryu, Y., Kobayashi, H., Guillaume, A., Moore, G., Ramakrishnan, L., Hendrix, V. C., Wolf, S., Sikka, M., Kiely, G., Wohlfahrt, G., Gielen, B., Roupsard, O., Toscano, P., Arain, A., & Cescatti, A. (2016). Global surface net-radiation at 5 km from MODIS Terra. Remote Sensing, 8, 739. https://api.semanticscholar.org/CorpusID:1517647

    Parameters:
        Rn (Union[Raster, np.ndarray, float]): Instantaneous net radiation (W/m²).
        hour_of_day (Union[Raster, np.ndarray, float]): Hour of the day (0-24).
        doy (Union[Raster, np.ndarray, float], optional): Day of the year (1-365).
        lat (Union[Raster, np.ndarray, float], optional): Latitude in degrees.
        sunrise_hour (Union[Raster, np.ndarray, float], optional): Hour of sunrise.
        daylight_hours (Union[Raster, np.ndarray, float], optional): Total daylight hours.

    Returns:
        Union[Raster, np.ndarray, float]: Daily net radiation (W/m²).
    """
    if daylight_hours is None or sunrise_hour is None and DOY is not None and lat is not None:
        sha_deg = SHA_deg_from_DOY_lat(DOY, lat)
        daylight_hours = daylight_from_SHA(sha_deg)
        sunrise_hour = sunrise_from_SHA(sha_deg)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        Rn_daily = 1.6 * Rn / (np.pi * np.sin(np.pi * (hour_of_day - sunrise_hour) / (daylight_hours)))
    
    return Rn_daily
