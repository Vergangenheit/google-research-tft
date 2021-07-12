sorgenia_farms = ['UP_PRCLCDPLRM_1', 'UP_PRCLCDMZRD_1', 'UP_PRCLCDPRZZ_1', 'UP_PRCLCMINEO_1', 'UP_PEPIZZA_1',
                  'UP_MPNTLCSMBC_1', 'UP_MPNTLCDMRN_1']

columns = ['plant_code', 'time', 'dew_point_2m_C', 'temperature_2m_C', 'msl_pressure_hPa', 'sfc_pressure_hPa',
           'precipitation_1h_mm', 'wind_speed_mean_10m_1h_ms',
           'wind_speed_mean_100m_1h_ms', 'wind_dir_mean_100m_1h_d', 'wind_dir_mean_10m_1h_d', 'wind_gusts_10m_1h_ms',
           'wind_gusts_10m_ms']

added_cols = ['hour', 'day_of_week', 'hours_from_start', 'categorical_id']

# query_observed1 = "SELECT meteomatics_weather.plant_code, meteomatics_weather.timestamp_utc, meteomatics_weather.dew_point_2m_c, " \
#                   "meteomatics_weather.temperature_2m_c, meteomatics_weather.msl_pressure_hpa, meteomatics_weather.sfc_pressure_hpa, " \
#                  "meteomatics_weather.precipitation_1h_mm, meteomatics_weather.wind_speed_mean_10m_1h_ms, meteomatics_weather.wind_speed_mean_100m_1h_ms," \
#                   " meteomatics_weather.wind_dir_mean_100m_1h_d, meteomatics_weather.wind_dir_mean_10m_1h_d, meteomatics_weather.wind_gusts_10m_1h_ms, " \
#                   "meteomatics_weather.wind_gusts_10m_ms FROM meteomatics_weather WHERE timestamp_utc between '{}' and '{}'"
#
# query_observed2 = "SELECT meteomatics_weather.plant_code, meteomatics_weather.timestamp_utc, meteomatics_weather.dew_point_2m_c, " \
#                   "meteomatics_weather.temperature_2m_c, meteomatics_weather.msl_pressure_hpa, meteomatics_weather.sfc_pressure_hpa, " \
#                  "meteomatics_weather.precipitation_1h_mm, meteomatics_weather.wind_speed_mean_10m_1h_ms, meteomatics_weather.wind_speed_mean_100m_1h_ms," \
#                   " meteomatics_weather.wind_dir_mean_100m_1h_d, meteomatics_weather.wind_dir_mean_10m_1h_d, meteomatics_weather.wind_gusts_10m_1h_ms, " \
#                   "meteomatics_weather.wind_gusts_10m_ms FROM meteomatics_forecast_weather WHERE forecast_timestamp_utc = '{}' and timestamp_query_utc = '{}'"
#
# query_fore = "SELECT meteomatics_weather.plant_code, meteomatics_weather.timestamp_utc, meteomatics_weather.dew_point_2m_c, " \
#                   "meteomatics_weather.temperature_2m_c, meteomatics_weather.msl_pressure_hpa, meteomatics_weather.sfc_pressure_hpa, " \
#                  "meteomatics_weather.precipitation_1h_mm, meteomatics_weather.wind_speed_mean_10m_1h_ms, meteomatics_weather.wind_speed_mean_100m_1h_ms," \
#                   " meteomatics_weather.wind_dir_mean_100m_1h_d, meteomatics_weather.wind_dir_mean_10m_1h_d, meteomatics_weather.wind_gusts_10m_1h_ms, " \
#                   "meteomatics_weather.wind_gusts_10m_ms FROM meteomatics_forecast_weather WHERE forecast_timestamp_utc between '{}' and '{}'"
