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
        Rn_Wm2: Union[Raster, np.ndarray, float],
        hour_of_day: Union[Raster, np.ndarray, float],
        day_of_year: Union[Raster, np.ndarray, float] = None,
        lat: Union[Raster, np.ndarray, float] = None,
        sunrise_hour: Union[Raster, np.ndarray, float] = None,
        daylight_hours: Union[Raster, np.ndarray, float] = None
        ) -> Union[Raster, np.ndarray, float]:
    """
    Integrate instantaneous net radiation (Rn) to daily average values using solar geometry parameters.

    This function estimates the daily average net radiation (W/m²) from instantaneous measurements, accounting for solar position and daylight duration. It supports Raster, numpy array, or float inputs for geospatial and scientific workflows. If sunrise time or daylight hours are not provided, they are calculated from day of year and latitude.

    Parameters:
        Rn_Wm2 (Union[Raster, np.ndarray, float]): Instantaneous net radiation (W/m²).
        hour_of_day (Union[Raster, np.ndarray, float]): Hour of the day (0-24) when Rn is measured.
        day_of_year (Union[Raster, np.ndarray, float], optional): Day of the year (1-365).
        lat (Union[Raster, np.ndarray, float], optional): Latitude in degrees.
        sunrise_hour (Union[Raster, np.ndarray, float], optional): Hour of sunrise (local time).
        daylight_hours (Union[Raster, np.ndarray, float], optional): Total daylight hours.

    Returns:
        Union[Raster, np.ndarray, float]: Daily average net radiation (W/m²).

    Notes:
        - To obtain total daily energy (J/m²), multiply the result by (daylight_hours * 3600).
        - If sunrise_hour or daylight_hours are not provided, they are computed from day_of_year and latitude using solar geometry.

    Reference:
        Verma, M., Fisher, J. B., Mallick, K., Ryu, Y., Kobayashi, H., Guillaume, A., Moore, G., Ramakrishnan, L., Hendrix, V. C., Wolf, S., Sikka, M., Kiely, G., Wohlfahrt, G., Gielen, B., Roupsard, O., Toscano, P., Arain, A., & Cescatti, A. (2016). Global surface net-radiation at 5 km from MODIS Terra. Remote Sensing, 8, 739. https://api.semanticscholar.org/CorpusID:1517647
    """
    if daylight_hours is None or sunrise_hour is None and day_of_year is not None and lat is not None:
        sha_deg = SHA_deg_from_DOY_lat(day_of_year, lat)
        daylight_hours = daylight_from_SHA(sha_deg)
        sunrise_hour = sunrise_from_SHA(sha_deg)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        Rn_daily = 1.6 * Rn_Wm2 / (np.pi * np.sin(np.pi * (hour_of_day - sunrise_hour) / (daylight_hours)))
    
    return Rn_daily
