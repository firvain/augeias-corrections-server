import os

import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine

load_dotenv('.env')
POSTGRESQL_URL = os.environ.get("POSTGRESQL_URL")
engine = create_engine(POSTGRESQL_URL)
sql = f"""with q as (select accuweather_direct.timestamp,
                  accuweather_direct.temp          as temp_accuweather,
                  accuweather_direct.rh            as humidity_accuweather,
                  accuweather_direct.precipitation as precipitation_accuweather,
                  accuweather_direct.wind_speed    as wind_speed_accuweather,
                  accuweather_direct.solar_irradiance solar_irradiance_accuweather,
                  a.temp                           as temp_accuweather_corrected,
                  a.rh                             as humidity_accuweather_corrected,
                  a.precipitation                  as precipitation_accuweather_corrected,
                  a.wind_speed                     as wind_speed_accuweather_corrected,
                  a.solar_irradiance               as solar_irradiance_corrected
           from accuweather_direct
                    inner join accuweather_corrected a on accuweather_direct.timestamp = a.timestamp)
select q.timestamp,
       q.temp_accuweather,
       q.humidity_accuweather,
       q.precipitation_accuweather,
       q.wind_speed_accuweather,
       q.solar_irradiance_accuweather,
       q.temp_accuweather_corrected,
       q.humidity_accuweather_corrected,
       q.precipitation_accuweather_corrected,
       q.wind_speed_accuweather_corrected,
       q.solar_irradiance_corrected,
       b."Air temperature" as temp_addvantage,
       b."RH"              as hummidity_addvantage,
       b."Precipitation" as precipitation_advantage,
       b."Wind speed 100 Hz" as wind_speed_addvantage,
       b."Pyranometer" as solar_irradiance_advantage

from q
         inner join addvantage b on q.timestamp = b.timestamp
order by b.timestamp asc"""

data = pd.read_sql(sql, engine)
data = data.set_index('timestamp')
data.index = pd.to_datetime(data.index).tz_convert('Europe/Athens')
temp = data[['temp_accuweather', 'temp_accuweather_corrected', 'temp_addvantage']]

mape = mean_absolute_error(temp['temp_accuweather'], temp['temp_addvantage'])*100
print(f"MAPE for temperature is {mape:.2f}")
mape_corrected = mean_absolute_error(temp['temp_accuweather_corrected'], temp['temp_addvantage'])*100
print(f"MAPE for temperature corrected is {mape_corrected:0.2f}")
# temp.plot()
# plt.savefig('temp.png')
humidity = data[['humidity_accuweather', 'humidity_accuweather_corrected', 'hummidity_addvantage']]
mape = mean_absolute_error(humidity['humidity_accuweather'], humidity['hummidity_addvantage'])*100
print(f"MAPE for humidity is {mape:.2f}")
mape_corrected = mean_absolute_error(humidity['humidity_accuweather_corrected'], humidity['hummidity_addvantage'])*100
print(f"MAPE for humidity corrected is {mape_corrected:0.2f}")

# humidity.plot()
# plt.savefig('humidity.png')
precipitation = data[['precipitation_accuweather', 'precipitation_accuweather_corrected', 'precipitation_advantage']]
mape = mean_absolute_error(precipitation['precipitation_accuweather'], precipitation['precipitation_advantage'])*100
print(f"MAPE for precipitation is {mape:.2f}")
mape_corrected = mean_absolute_error(precipitation['precipitation_accuweather_corrected'], precipitation['precipitation_advantage'])*100
print(f"MAPE for precipitation corrected is {mape_corrected:0.2f}")

# precipitation.plot()
# plt.savefig('precipitation.png')
wind_speed = data[['wind_speed_accuweather', 'wind_speed_accuweather_corrected', 'wind_speed_addvantage']]
mape = mean_absolute_error(wind_speed['wind_speed_accuweather'], wind_speed['wind_speed_addvantage'])*100
print(f"MAPE for wind speed is {mape:0.2f}")
mape_corrected = mean_absolute_error(wind_speed['wind_speed_accuweather_corrected'], wind_speed['wind_speed_addvantage'])*100
print(f"MAPE for wind speed corrected is {mape_corrected:0.2f}")
# wind_speed.plot()
# plt.savefig('wind_speed.png')
# solar_irradiance = data[['solar_irradiance_accuweather', 'solar_irradiance_corrected', 'solar_irradiance_advantage']]
# solar_irradiance.plot()
# plt.savefig('solar_irradiance.png')


