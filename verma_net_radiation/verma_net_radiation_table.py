"""
Verma Net Radiation Table Utilities
===================================

This module provides a function to process tabular (DataFrame) inputs for the Verma net radiation model.
It computes net radiation and its components for each row of the input DataFrame and appends the results as new columns.

Key Features:
-------------
- Accepts a pandas DataFrame with required input columns for net radiation calculation.
- Computes outgoing/incoming shortwave and longwave radiation, and net radiation.
- Returns a DataFrame with additional columns for all calculated radiation components.

Reference:
----------
Verma, M., Fisher, J. B., Mallick, K., Ryu, Y., Kobayashi, H., Guillaume, A., Moore, G., Ramakrishnan, L., Hendrix, V. C., Wolf, S., Sikka, M., Kiely, G., Wohlfahrt, G., Gielen, B., Roupsard, O., Toscano, P., Arain, A., & Cescatti, A. (2016). Global surface net-radiation at 5 km from MODIS Terra. Remote Sensing, 8, 739. https://api.semanticscholar.org/CorpusID:1517647

Example Usage:
--------------
>>> from verma_net_radiation_table import verma_net_radiation_table
>>> import pandas as pd
>>> df = pd.read_csv('inputs.csv')
>>> df_out = verma_net_radiation_table(df)
"""
import numpy as np
from pandas import DataFrame
from .model import verma_net_radiation

def verma_net_radiation_table(verma_net_radiation_inputs_df: DataFrame) -> DataFrame:
    """
    Process a DataFrame containing inputs for Verma net radiation calculations.

    This function takes a DataFrame with columns representing various input parameters
    required for calculating net radiation and its components. It processes the inputs,
    computes the radiation components using the `verma_net_radiation` function,
    and appends the results as new columns to the input DataFrame.

    Parameters:
        verma_net_radiation_inputs_df (DataFrame): A DataFrame containing the following columns:
            - Rg: Incoming shortwave radiation (W/m²).
            - albedo: Surface albedo (unitless, constrained between 0 and 1).
            - ST_C: Surface temperature in Celsius.
            - emissivity: Surface emissivity (unitless, constrained between 0 and 1).
            - Ta_C: Air temperature in Celsius.
            - RH: Relative humidity (fractional, e.g., 0.5 for 50%).

    Returns:
        DataFrame: A copy of the input DataFrame with additional columns for the calculated
        radiation components:
            - SWout: Outgoing shortwave radiation (W/m²).
            - LWin: Incoming longwave radiation (W/m²).
            - LWout: Outgoing longwave radiation (W/m²).
            - Rn: Instantaneous net radiation (W/m²).
    """

    # Extract and convert each required input column to a numpy array for computation.
    # This ensures compatibility with the underlying model functions and vectorized operations.
    SWin = np.array(verma_net_radiation_inputs_df.Rg)  # Incoming shortwave radiation
    albedo = np.array(verma_net_radiation_inputs_df.albedo)  # Surface albedo
    ST_C = np.array(verma_net_radiation_inputs_df.ST_C)  # Surface temperature (Celsius)
    # Note: The input column for emissivity may be named 'EmisWB' in the DataFrame.
    # If so, use that column; otherwise, fall back to 'emissivity' if present.
    if 'EmisWB' in verma_net_radiation_inputs_df.columns:
        emissivity = np.array(verma_net_radiation_inputs_df.EmisWB)
    else:
        emissivity = np.array(verma_net_radiation_inputs_df.emissivity)
    Ta_C = np.array(verma_net_radiation_inputs_df.Ta_C)  # Air temperature (Celsius)
    RH = np.array(verma_net_radiation_inputs_df.RH)  # Relative humidity (fractional)

    # Call the main model function to compute all radiation components.
    # The function returns a dictionary with keys for each component.
    results = verma_net_radiation(
        SWin=SWin,
        albedo=albedo,
        ST_C=ST_C,
        emissivity=emissivity,
        Ta_C=Ta_C,
        RH=RH,
    )

    # Create a copy of the input DataFrame to avoid modifying the original.
    verma_net_radiation_outputs_df = verma_net_radiation_inputs_df.copy()

    # Add each calculated radiation component as a new column in the output DataFrame.
    for key, value in results.items():
        verma_net_radiation_outputs_df[key] = value

    # Return the DataFrame with appended results.
    return verma_net_radiation_outputs_df
