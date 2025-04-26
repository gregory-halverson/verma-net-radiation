from typing import Union, Dict
import warnings
import numpy as np
import rasters as rt
from rasters import Raster

from sun_angles import daylight_from_SHA
from sun_angles import sunrise_from_SHA
from sun_angles import solar_dec_deg_from_day_angle_rad
from sun_angles import day_angle_rad_from_DOY
from sun_angles import SHA_deg_from_DOY_lat

STEFAN_BOLTZMAN_CONSTANT = 5.67036713e-8  # SI units watts per square meter per kelvin to the fourth

def process_verma_net_radiation(
        SWin: Union[Raster, np.ndarray],
        albedo: Union[Raster, np.ndarray],
        ST_C: Union[Raster, np.ndarray],
        emissivity: Union[Raster, np.ndarray],
        Ta_C: Union[Raster, np.ndarray],
        RH: Union[Raster, np.ndarray],
        cloud_mask: Union[Raster, np.ndarray] = None) -> Dict[str, Union[Raster, np.ndarray]]:
    """
    Calculate instantaneous net radiation and its components.

    Parameters:
        SWin (np.ndarray): Incoming shortwave radiation (W/m²).
        albedo (np.ndarray): Surface albedo (unitless, constrained between 0 and 1).
        ST_C (np.ndarray): Surface temperature in Celsius.
        emissivity (np.ndarray): Surface emissivity (unitless, constrained between 0 and 1).
        Ta_C (np.ndarray): Air temperature in Celsius.
        RH (np.ndarray): Relative humidity (fractional, e.g., 0.5 for 50%).
        cloud_mask (np.ndarray, optional): Boolean mask indicating cloudy areas (True for cloudy).

    Returns:
        Dict: A dictionary containing:
            - "SWout": Outgoing shortwave radiation (W/m²).
            - "LWin": Incoming longwave radiation (W/m²).
            - "LWout": Outgoing longwave radiation (W/m²).
            - "Rn": Instantaneous net radiation (W/m²).
    """
    results = {}

    # Convert surface temperature from Celsius to Kelvin
    ST_K = ST_C + 273.15

    # Convert air temperature from Celsius to Kelvin
    Ta_K = Ta_C + 273.15

    # Calculate water vapor pressure in Pascals using air temperature and relative humidity
    Ea_Pa = (RH * 0.6113 * (10 ** (7.5 * (Ta_K - 273.15) / (Ta_K - 35.85)))) * 1000
    
    # Constrain albedo between 0 and 1
    albedo = np.clip(albedo, 0, 1)

    # Calculate outgoing shortwave from incoming shortwave and albedo
    SWout = np.clip(SWin * albedo, 0, None)
    results["SWout"] = SWout

    # Calculate instantaneous net radiation from components
    SWnet = np.clip(SWin - SWout, 0, None)

    # Calculate atmospheric emissivity
    eta1 = 0.465 * Ea_Pa / Ta_K
    eta2 = -(1.2 + 3 * eta1) ** 0.5
    eta2 = eta2.astype(float)
    eta3 = np.exp(eta2)
    atmospheric_emissivity = np.where(eta2 != 0, (1 - (1 + eta1) * eta3), np.nan)

    if cloud_mask is None:
        # Calculate incoming longwave for clear sky
        LWin = atmospheric_emissivity * STEFAN_BOLTZMAN_CONSTANT * Ta_K ** 4
    else:
        # Calculate incoming longwave for clear sky and cloudy
        LWin = np.where(
            ~cloud_mask,
            atmospheric_emissivity * STEFAN_BOLTZMAN_CONSTANT * Ta_K ** 4,
            STEFAN_BOLTZMAN_CONSTANT * Ta_K ** 4
        )
    
    results["LWin"] = LWin

    # Constrain emissivity between 0 and 1
    emissivity = np.clip(emissivity, 0, 1)

    # Calculate outgoing longwave from land surface temperature and emissivity
    LWout = emissivity * STEFAN_BOLTZMAN_CONSTANT * ST_K ** 4
    results["LWout"] = LWout

    # Calculate net longwave radiation
    LWnet = np.clip(LWin - LWout, 0, None)

    # Constrain negative values of instantaneous net radiation
    Rn = np.clip(SWnet + LWnet, 0, None)
    results["Rn"] = Rn

    return results

def daily_Rn_integration_verma(
        Rn: Union[Raster, np.ndarray],
        hour_of_day: Union[Raster, np.ndarray],
        doy: Union[Raster, np.ndarray] = None,
        lat: Union[Raster, np.ndarray] = None,
        sunrise_hour: Union[Raster, np.ndarray] = None,
        daylight_hours: Union[Raster, np.ndarray] = None) -> Raster:
    """
    Calculate daily net radiation using solar parameters.

    This represents the average rate of energy transfer from sunrise to sunset
    in watts per square meter. To get the total energy transferred, multiply
    by the number of seconds in the daylight period (daylight_hours * 3600).

    Parameters:
        Rn (Union[Raster, np.ndarray]): Instantaneous net radiation (W/m²).
        hour_of_day (Union[Raster, np.ndarray]): Hour of the day (0-24).
        doy (Union[Raster, np.ndarray], optional): Day of the year (1-365).
        lat (Union[Raster, np.ndarray], optional): Latitude in degrees.
        sunrise_hour (Union[Raster, np.ndarray], optional): Hour of sunrise.
        daylight_hours (Union[Raster, np.ndarray], optional): Total daylight hours.

    Returns:
        Raster: Daily net radiation (W/m²).
    """
    if daylight_hours is None or sunrise_hour is None and doy is not None and lat is not None:
        sha_deg = SHA_deg_from_DOY_lat(doy, lat)
        daylight_hours = daylight_from_SHA(sha_deg)
        sunrise_hour = sunrise_from_SHA(sha_deg)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        Rn_daily = 1.6 * Rn / (np.pi * np.sin(np.pi * (hour_of_day - sunrise_hour) / (daylight_hours)))
    
    return Rn_daily