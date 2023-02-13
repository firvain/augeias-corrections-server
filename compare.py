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
order by b.timestamp desc limit 24"""

data = pd.read_sql(sql, engine)
data = data.set_index('timestamp')
data.index = pd.to_datetime(data.index).tz_convert('Europe/Athens')

mape_array = []
temp = data[['temp_accuweather', 'temp_accuweather_corrected', 'temp_addvantage']]

mape = mean_absolute_error(temp['temp_accuweather'], temp['temp_addvantage'])*100

print(f"MAPE for temperature is {mape:.2f}")
mape_corrected = mean_absolute_error(temp['temp_accuweather_corrected'], temp['temp_addvantage'])*100
print(f"MAPE for temperature corrected is {mape_corrected:0.2f}")
mape_array.append([mape, mape_corrected])

plt.figure(figsize=(20, 10))
plt.title('Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.plot(temp.index, temp['temp_accuweather'], label='Accuweather')
plt.plot(temp.index, temp['temp_accuweather_corrected'], label='Accuweather corrected')
plt.plot(temp.index, temp['temp_addvantage'], label='Addvantage')
plt.legend()
plt.tight_layout()
# plt.savefig('Z:\Projects\ΑΥΓΕΙΑΣ\ΕΓΓΡΑΦΑ/temp_accuweather_addvantage.png', dpi=300)
plt.show()


humidity = data[['humidity_accuweather', 'humidity_accuweather_corrected', 'hummidity_addvantage']]
mape = mean_absolute_error(humidity['humidity_accuweather'], humidity['hummidity_addvantage'])*100
print(f"MAPE for humidity is {mape:.2f}")
mape_corrected = mean_absolute_error(humidity['humidity_accuweather_corrected'], humidity['hummidity_addvantage'])*100
print(f"MAPE for humidity corrected is {mape_corrected:0.2f}")
mape_array.append([mape, mape_corrected])
plt.figure(figsize=(20, 10))
plt.title('Humidity')
plt.xlabel('Time')
plt.ylabel('Humidity')
plt.plot(humidity.index, humidity['humidity_accuweather'], label='Accuweather')
plt.plot(humidity.index, humidity['humidity_accuweather_corrected'], label='Accuweather corrected')
plt.plot(humidity.index, humidity['hummidity_addvantage'], label='Addvantage')
plt.legend()
plt.tight_layout()
# plt.savefig('Z:\Projects\ΑΥΓΕΙΑΣ\ΕΓΓΡΑΦΑ\humidity_accuweather_addvantage.png', dpi=300)
plt.show()

precipitation = data[['precipitation_accuweather', 'precipitation_accuweather_corrected', 'precipitation_advantage']]
mape = mean_absolute_error(precipitation['precipitation_accuweather'], precipitation['precipitation_advantage'])*100
print(f"MAPE for precipitation is {mape:.2f}")
mape_corrected = mean_absolute_error(precipitation['precipitation_accuweather_corrected'], precipitation['precipitation_advantage'])*100
print(f"MAPE for precipitation corrected is {mape_corrected:0.2f}")
mape_array.append([mape, mape_corrected])
plt.figure(figsize=(20, 10))
plt.title('Precipitation')
plt.xlabel('Time')
plt.ylabel('Precipitation')
plt.plot(precipitation.index, precipitation['precipitation_accuweather'], label='Accuweather')
plt.plot(precipitation.index, precipitation['precipitation_accuweather_corrected'], label='Accuweather corrected')
plt.plot(precipitation.index, precipitation['precipitation_advantage'], label='Addvantage')
plt.legend()
plt.tight_layout()
# plt.savefig('Z:\Projects\ΑΥΓΕΙΑΣ\ΕΓΓΡΑΦΑ\precipitation_accuweather_addvantage.png', dpi=300)
plt.show()

wind_speed = data[['wind_speed_accuweather', 'wind_speed_accuweather_corrected', 'wind_speed_addvantage']]
mape = mean_absolute_error(wind_speed['wind_speed_accuweather'], wind_speed['wind_speed_addvantage'])*100
print(f"MAPE for wind speed is {mape:0.2f}")
mape_corrected = mean_absolute_error(wind_speed['wind_speed_accuweather_corrected'], wind_speed['wind_speed_addvantage'])*100
print(f"MAPE for wind speed corrected is {mape_corrected:0.2f}")
mape_array.append([mape, mape_corrected])
plt.figure(figsize=(20, 10))
plt.title('Wind speed')
plt.xlabel('Time')
plt.ylabel('Wind speed')
plt.plot(wind_speed.index, wind_speed['wind_speed_accuweather'], label='Accuweather')
plt.plot(wind_speed.index, wind_speed['wind_speed_accuweather_corrected'], label='Accuweather corrected')
plt.plot(wind_speed.index, wind_speed['wind_speed_addvantage'], label='Addvantage')
plt.legend()
plt.tight_layout()
# plt.savefig('Z:\Projects\ΑΥΓΕΙΑΣ\ΕΓΓΡΑΦΑ\wind_speed_accuweather_addvantage.png', dpi=300)
plt.show()


data['solar_irradiance_corrected'] = data['solar_irradiance_corrected'].apply(lambda x: x if x > 0 else 0)
data['solar_irradiance_accuweather'] = data['solar_irradiance_accuweather'].apply(lambda x: x if x > 0 else 0)
data['solar_irradiance_advantage'] = data['solar_irradiance_advantage'].apply(lambda x: x if x > 0 else 0)
solar_irradiance = data[['solar_irradiance_accuweather', 'solar_irradiance_corrected', 'solar_irradiance_advantage']]
mape = mean_absolute_error(solar_irradiance['solar_irradiance_accuweather'], solar_irradiance['solar_irradiance_advantage'])*100
print(f"MAPE for solar irradiance is {mape:0.2f}")
mape_corrected = mean_absolute_error(solar_irradiance['solar_irradiance_corrected'], solar_irradiance['solar_irradiance_advantage'])*100
print(f"MAPE for solar irradiance corrected is {mape_corrected:0.2f}")
mape_array.append([mape, mape_corrected])

plt.figure(figsize=(20, 10))
plt.title('Solar irradiance')
plt.xlabel('Time')
plt.ylabel('Solar irradiance')
plt.plot(solar_irradiance.index, solar_irradiance['solar_irradiance_accuweather'], label='Accuweather')
plt.plot(solar_irradiance.index, solar_irradiance['solar_irradiance_corrected'], label='Accuweather corrected')
plt.plot(solar_irradiance.index, solar_irradiance['solar_irradiance_advantage'], label='Addvantage')
plt.legend()
plt.tight_layout()
# plt.savefig('Z:\Projects\ΑΥΓΕΙΑΣ\ΕΓΓΡΑΦΑ\solar_irradiance_accuweather_addvantage.png', dpi=300)
plt.show()



mape_df = pd.DataFrame(mape_array)


mape_df.index = ['Temperature', 'Humidity', 'Precipitation', 'Wind speed', 'Solar irradiance']

mape_df.columns = ['MAPE', 'MAPE corrected']
mape_df.drop('Precipitation', inplace=True, axis=0)


mape_df.plot.bar(figsize=(20, 10), rot=0, )

plt.ylabel('MAPE')
plt.legend()
plt.tight_layout()
# plt.savefig('Z:\Projects\ΑΥΓΕΙΑΣ\ΕΓΓΡΑΦΑ\mape_accuweather_addvantage.png', dpi=300)
plt.show()


mape_df['Percentage decrease'] = (mape_df['MAPE'] - mape_df['MAPE corrected'])/mape_df['MAPE']*100
# mape_df['Percentage decrease'] = mape_df['Percentage decrease'].apply(lambda x: f"{x:0.2f}%")
mape_df.drop('MAPE', inplace=True, axis=1)
mape_df.drop('MAPE corrected', inplace=True, axis=1)
ax = mape_df.plot.bar(figsize=(20, 10), rot=0 )
for container in ax.containers:
    ax.bar_label(container , fmt='%.2f%%', fontsize=14)
plt.title('Percentage decrease in MAPE')
plt.ylabel('Percentage decrease')
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.plot()
# plt.savefig('Z:\Projects\ΑΥΓΕΙΑΣ\ΕΓΓΡΑΦΑ\percentage_decrease_accuweather_addvantage.png', dpi=300)
plt.show()


