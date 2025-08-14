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

from datetime import datetime
from typing import Union
import warnings
from geopandas import GeoSeries
import numpy as np
from dateutil import parser

from rasters import Raster
from rasters import SpatialGeometry

from solar_apparent_time import calculate_solar_day_of_year, calculate_solar_hour_of_day
from sun_angles import daylight_from_SHA, sunrise_from_SHA, SHA_deg_from_DOY_lat

def daily_Rn_integration_verma(
        Rn_Wm2: Union[Raster, np.ndarray, float],
        time_UTC: Union[datetime, str, list, np.ndarray] = None,
        geometry: Union[SpatialGeometry, GeoSeries] = None,
        hour_of_day: Union[Raster, np.ndarray, float] = None,
        day_of_year: Union[Raster, np.ndarray, int] = None,
        lat: Union[Raster, np.ndarray, float] = None,
        lon: Union[Raster, np.ndarray, float] = None,
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
    # If latitude is not provided, try to extract from geometry
    if lat is None and isinstance(geometry, SpatialGeometry):
        lat = geometry.lat
    elif lat is None and isinstance(geometry, GeoSeries):
        lat = geometry.y

    if lon is None and isinstance(geometry, SpatialGeometry):
        lon = geometry.lon
    elif lon is None and isinstance(geometry, GeoSeries):
        lon = geometry.x

    # Handle day_of_year input: convert lists to np.ndarray
    if day_of_year is not None:
        if isinstance(day_of_year, list):
            day_of_year = np.array(day_of_year)

    # Handle lat input: convert lists to np.ndarray
    if lat is not None:
        if isinstance(lat, list):
            lat = np.array(lat)

    # print(type(time_UTC), time_UTC)

    # If day_of_year is not provided, try to infer from time_UTC
    if day_of_year is None and time_UTC is not None:
        # Handle string or list of strings for time_UTC
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)
        elif isinstance(time_UTC, list):
            time_UTC = [parser.parse(t) if isinstance(t, str) else t for t in time_UTC]
        elif isinstance(time_UTC, np.ndarray) and time_UTC.dtype.type is np.str_:
            time_UTC = np.array([parser.parse(t) for t in time_UTC])

        day_of_year = calculate_solar_day_of_year(
            time_UTC=time_UTC,
            geometry=geometry,
            lat=lat,
            lon=lon
        )    

    # If hour_of_day is not provided, try to calculate from time_UTC and geometry
    if hour_of_day is None and time_UTC is not None:
        # Convert pandas series to numpy arrays to avoid broadcasting issues
        lat_array = np.asarray(lat) if lat is not None else None
        lon_array = np.asarray(lon) if lon is not None else None
        
        hour_of_day = calculate_solar_hour_of_day(
            time_UTC=time_UTC,
            geometry=None,
            lat=lat_array,
            lon=lon_array
        )
        
        # If result is 2D (still a broadcasting issue in solar-apparent-time), extract diagonal
        if hasattr(hour_of_day, 'shape') and len(hour_of_day.shape) == 2:
            hour_of_day = np.diag(hour_of_day)

    # Calculate sunrise and daylight hours if not provided
    if (daylight_hours is None or sunrise_hour is None) and day_of_year is not None and lat is not None:
        sha_deg = SHA_deg_from_DOY_lat(day_of_year, lat)
        if daylight_hours is None:
            daylight_hours = daylight_from_SHA(sha_deg)
        if sunrise_hour is None:
            sunrise_hour = sunrise_from_SHA(sha_deg)

    # Validate that we have all required parameters
    if hour_of_day is None or sunrise_hour is None or daylight_hours is None:
        warnings.warn("Could not calculate all required solar parameters. Returning NaN values.")
        if hasattr(Rn_Wm2, 'shape'):
            return np.full_like(Rn_Wm2, np.nan)
        else:
            return np.nan

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Calculate daily net radiation
        denominator = np.pi * np.sin(np.pi * (hour_of_day - sunrise_hour) / daylight_hours)
        
        # Convert to numpy arrays to avoid pandas broadcasting issues
        Rn_array = np.asarray(Rn_Wm2)
        denom_array = np.asarray(denominator)
        
        # Handle cases where denominator is zero or negative (invalid solar geometry)
        if hasattr(denom_array, 'shape'):
            # Array case
            mask = (denom_array <= 0) | np.isnan(denom_array) | np.isinf(denom_array)
            Rn_daily = 1.6 * Rn_array / denom_array
            Rn_daily = np.where(mask, np.nan, Rn_daily)
        else:
            # Scalar case
            if denom_array <= 0 or np.isnan(denom_array) or np.isinf(denom_array):
                Rn_daily = np.nan
            else:
                Rn_daily = 1.6 * Rn_array / denom_array
    
    return Rn_daily
