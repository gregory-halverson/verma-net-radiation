
from typing import Union, Dict
import numpy as np
import warnings
from rasters import Raster

from .brutsaert_atmospheric_emissivity import brutsaert_atmospheric_emissivity
from .incoming_longwave_radiation import incoming_longwave_radiation
from .outgoing_longwave_radiation import outgoing_longwave_radiation

STEFAN_BOLTZMAN_CONSTANT = 5.67036713e-8  # SI units watts per square meter per kelvin to the fourth

def verma_net_radiation(
        ST_C: Union[Raster, np.ndarray, float],
        emissivity: Union[Raster, np.ndarray, float],
        albedo: Union[Raster, np.ndarray, float],
        SWin: Union[Raster, np.ndarray, float], 
        Ta_C: Union[Raster, np.ndarray, float],
        RH: Union[Raster, np.ndarray, float],
        cloud_mask: Union[Raster, np.ndarray, float, None] = None
        ) -> Dict[str, Union[Raster, np.ndarray, float]]:
    """
    Calculate instantaneous net radiation and its components.

    This function implements the net radiation and component fluxes as described in:
    Verma, M., Fisher, J. B., Mallick, K., Ryu, Y., Kobayashi, H., Guillaume, A., Moore, G., Ramakrishnan, L., Hendrix, V. C., Wolf, S., Sikka, M., Kiely, G., Wohlfahrt, G., Gielen, B., Roupsard, O., Toscano, P., Arain, A., & Cescatti, A. (2016). Global surface net-radiation at 5 km from MODIS Terra. Remote Sensing, 8, 739. https://api.semanticscholar.org/CorpusID:1517647

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

    # Calculate atmospheric emissivity using Brutsaert (1975) model
    atmospheric_emissivity = brutsaert_atmospheric_emissivity(Ea_Pa, Ta_K)

    # Calculate incoming longwave radiation (clear/cloudy)
    LWin = incoming_longwave_radiation(atmospheric_emissivity, Ta_K, cloud_mask)
    
    results["LWin"] = LWin

    # Constrain emissivity between 0 and 1
    emissivity = np.clip(emissivity, 0, 1)

    # Calculate outgoing longwave from land surface temperature and emissivity
    LWout = outgoing_longwave_radiation(emissivity, ST_K)
    results["LWout"] = LWout

    # Calculate net longwave radiation
    LWnet = LWin - LWout

    # Constrain negative values of instantaneous net radiation
    Rn = np.clip(SWnet + LWnet, 0, None)
    results["Rn"] = Rn

    return results
