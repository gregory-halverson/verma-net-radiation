# Net Radiation and Daily Upscaling Remote Sensing in Python

This Python package implements the net radiation and daily upscaling methods described in Verma et al 2016.

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
Lead developer<br>
NASA Jet Propulsion Laboratory 329G

## Installation

This package is distributed using the pip package manager as `verma-net-radiation` with dashes.

```
pip install verma-net-radiation
```

## Usage

Import this package as `verma_net_radiation` with underscores.

This module provides functions to calculate instantaneous net radiation and its components, integrate daily net radiation, and process radiation data from a DataFrame. Below is a detailed explanation of each function and how to use them.

### `verma_net_radiation`

**Description**:  
Calculates instantaneous net radiation and its components based on input parameters.

**Parameters**:
- `SWin` (Union[Raster, np.ndarray, float]): Incoming shortwave radiation (W/m²).
- `albedo` (Union[Raster, np.ndarray, float]): Surface albedo (unitless, constrained between 0 and 1).
- `ST_C` (Union[Raster, np.ndarray, float]): Surface temperature in Celsius.
- `emissivity` (Union[Raster, np.ndarray, float]): Surface emissivity (unitless, constrained between 0 and 1).
- `Ta_C` (Union[Raster, np.ndarray, float]): Air temperature in Celsius.
- `RH` (Union[Raster, np.ndarray, float]): Relative humidity (fractional, e.g., 0.5 for 50%).
- `cloud_mask` (Union[Raster, np.ndarray, float], optional): Boolean mask indicating cloudy areas (True for cloudy).

**Returns**:
A dictionary containing:
- `"SWout"`: Outgoing shortwave radiation (W/m²).
- `"LWin"`: Incoming longwave radiation (W/m²).
- `"LWout"`: Outgoing longwave radiation (W/m²).
- `"Rn"`: Instantaneous net radiation (W/m²).

**Example**:
```python
results = verma_net_radiation(
    SWin=SWin_array,
    albedo=albedo_array,
    ST_C=surface_temp_array,
    emissivity=emissivity_array,
    Ta_C=air_temp_array,
    RH=relative_humidity_array,
    cloud_mask=cloud_mask_array
)
```

### `daily_Rn_integration_verma`

**Description**:  
Calculates daily net radiation using solar parameters.

**Parameters**:
- `Rn` (Union[Raster, np.ndarray]): Instantaneous net radiation (W/m²).
- `hour_of_day` (Union[Raster, np.ndarray]): Hour of the day (0-24).
- `doy` (Union[Raster, np.ndarray], optional): Day of the year (1-365).
- `lat` (Union[Raster, np.ndarray], optional): Latitude in degrees.
- `sunrise_hour` (Union[Raster, np.ndarray], optional): Hour of sunrise.
- `daylight_hours` (Union[Raster, np.ndarray], optional): Total daylight hours.

**Returns**:
- `Raster`: Daily net radiation (W/m²).

**Example**:
```python
daily_Rn = daily_Rn_integration_verma(
    Rn=Rn_array,
    hour_of_day=hour_of_day_array,
    doy=day_of_year_array,
    lat=latitude_array,
    sunrise_hour=sunrise_hour_array,
    daylight_hours=daylight_hours_array
)
```

---

### `verma_net_radiation_table`

**Description**:  
Processes a DataFrame containing inputs for Verma net radiation calculations and appends the results as new columns.

**Parameters**:
- `verma_net_radiation_inputs_df` (DataFrame): A DataFrame containing the following columns:
  - `Rg`: Incoming shortwave radiation (W/m²).
  - `albedo`: Surface albedo (unitless, constrained between 0 and 1).
  - `ST_C`: Surface temperature in Celsius.
  - `EmisWB` or `emissivity`: Surface emissivity (unitless, constrained between 0 and 1).
  - `Ta_C`: Air temperature in Celsius.
  - `RH`: Relative humidity (fractional, e.g., 0.5 for 50%).

**Returns**:
- `DataFrame`: A copy of the input DataFrame with additional columns for the calculated radiation components:
  - `SWout`: Outgoing shortwave radiation (W/m²).
  - `LWin`: Incoming longwave radiation (W/m²).
  - `LWout`: Outgoing longwave radiation (W/m²).
  - `Rn`: Instantaneous net radiation (W/m²).

**Example**:
```python
output_df = verma_net_radiation_table(input_df)
```

## References

**Brutsaert, W. (1975).** On a Derivable Formula for Long‐Wave Radiation from Clear Skies. *Water Resources Research, 11*(5), 742–744. https://doi.org/10.1029/WR011i005p00742  
*Empirical model for clear-sky atmospheric emissivity, used in the calculation of incoming longwave radiation.*

**Liou, K. N. (2002).** *An Introduction to Atmospheric Radiation* (2nd ed.). Academic Press. (See Eq. 2.3.1)  
*Textbook reference for the Stefan-Boltzmann Law and radiative transfer theory, used for outgoing longwave radiation calculations.*

**Stefan, J. (1879).** Über die Beziehung zwischen der Wärmestrahlung und der Temperatur. *Sitzungsberichte der mathematisch-naturwissenschaftlichen Classe der kaiserlichen Akademie der Wissenschaften*, 79, 391–428.  
*Original publication of the Stefan-Boltzmann Law, fundamental to blackbody radiation calculations.*

**Verma, M., Fisher, J. B., Mallick, K., Ryu, Y., Kobayashi, H., Guillaume, A., Moore, G., Ramakrishnan, L., Hendrix, V. C., Wolf, S., Sikka, M., Kiely, G., Wohlfahrt, G., Gielen, B., Roupsard, O., Toscano, P., Arain, A., & Cescatti, A. (2016).** Global surface net-radiation at 5 km from MODIS Terra. *Remote Sensing, 8*, 739. [Link](https://api.semanticscholar.org/CorpusID:1517647)  
*Primary methodology for net radiation and daily upscaling as implemented in this package.*
