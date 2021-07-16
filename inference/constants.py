sorgenia_farms = ['UP_PRCLCDPLRM_1', 'UP_PRCLCDMZRD_1', 'UP_PRCLCDPRZZ_1', 'UP_PRCLCMINEO_1', 'UP_PEPIZZA_1',
                  'UP_MPNTLCSMBC_1', 'UP_MPNTLCDMRN_1']

columns = ['plant_code', 'time', 'kwh', 'dew_point_2m_C', 'temperature_2m_C', 'msl_pressure_hPa', 'sfc_pressure_hPa',
           'precipitation_1h_mm', 'wind_speed_mean_10m_1h_ms',
           'wind_speed_mean_100m_1h_ms', 'wind_dir_mean_100m_1h_d', 'wind_dir_mean_10m_1h_d', 'wind_gusts_10m_1h_ms',
           'wind_gusts_10m_ms']

added_cols = ['hour', 'day_of_week', 'hours_from_start', 'categorical_id']

create_table = """
                create table ml_predictions (
	forecast_date_utc timestamp,
	plant_name_up varchar(30),
	t_1 numeric,
	t_2 numeric,
	t_3 numeric,
	t_4 numeric,
	t_5 numeric,
	t_6 numeric,
	t_7 numeric,
	t_8 numeric,
	t_9 numeric,
	t_10 numeric,
	t_11 numeric,
	t_12 numeric,
	primary key (forecast_date_utc, plant_name_up)

);"""

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

preds_query: str = "select * from {} where extract(month from forecast_time_utc) between extract(month from NOW()) - {} and extract(month from NOW())"

preds_query_interim: str = "select * from {} where extract(month from forecast_time_utc) between extract(month from TO_TIMESTAMP('2020-12-31 10:00:00', 'YYYY-MM-DD HH:MI:SS')) - {} and extract(month from TO_TIMESTAMP('2020-12-31 10:00:00', 'YYYY-MM-DD HH:MI:SS'))"

query_energy: str = "SELECT * FROM {} WHERE start_date_utc >= '{}' and start_date_utc < to_timestamp('{}', 'YYYY-MM-DD HH24:MI:SS') + '1 hour'::interval;"