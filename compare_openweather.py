import os

import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine

load_dotenv('.env')
POSTGRESQL_URL = os.environ.get("POSTGRESQL_URL")
engine = create_engine(POSTGRESQL_URL)
sql = f"""with q as (select openweather_direct.timestamp,
                  openweather_direct.temp       as temp_openweather,
                  openweather_direct.humidity   as humidity_openweather,
                  openweather_direct.wind_speed as wind_speed_openweather,

                  a.temp                        as temp_openweather_corrected,
                  a.rh                          as humidity_openweather_corrected,
                  a.wind_speed                  as wind_speed_openweather_corrected

           from openweather_direct
                    inner join openweather_corrected_test a on openweather_direct.timestamp = a.timestamp)
select q.timestamp,
       q.temp_openweather,
       q.humidity_openweather,
       q.wind_speed_openweather,
       q.temp_openweather_corrected,
       q.humidity_openweather_corrected,
       q.wind_speed_openweather_corrected,
       b."Air temperature"   as temp_addvantage,
       b."RH"                as hummidity_addvantage,
       b."Wind speed 100 Hz" as wind_speed_addvantage

from q
         inner join addvantage b on q.timestamp = b.timestamp
order by timestamp desc
limit 12"""

data = pd.read_sql(sql, engine)
data = data.set_index('timestamp')
data.to_csv('test.csv')

data.index = pd.to_datetime(data.index).tz_convert('Europe/Athens')
mape_array = []
temp = data[['temp_openweather', 'temp_openweather_corrected', 'temp_addvantage']]

mape = mean_absolute_error(temp['temp_openweather'], temp['temp_addvantage'])*100

print(f"MAPE for temperature is {mape:.2f}")
mape_corrected = mean_absolute_error(temp['temp_openweather_corrected'], temp['temp_addvantage'])*100
print(f"MAPE for temperature corrected is {mape_corrected:0.2f}")
mape_array.append([mape, mape_corrected])

plt.figure(figsize=(20, 10))
plt.title('Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.plot(temp.index, temp['temp_openweather'], label='Accuweather')
plt.plot(temp.index, temp['temp_openweather_corrected'], label='Openweather corrected')
plt.plot(temp.index, temp['temp_addvantage'], label='Addvantage')
plt.legend()
plt.tight_layout()
plt.savefig('Z:\Projects\ΑΥΓΕΙΑΣ\ΕΓΓΡΑΦΑ/temp_openweather_addvantage.png', dpi=300)
plt.show()

humidity = data[['humidity_openweather', 'humidity_openweather_corrected', 'hummidity_addvantage']]
mape = mean_absolute_error(humidity['humidity_openweather'], humidity['hummidity_addvantage'])*100
print(f"MAPE for humidity is {mape:.2f}")
mape_corrected = mean_absolute_error(humidity['humidity_openweather_corrected'], humidity['hummidity_addvantage'])*100
print(f"MAPE for humidity corrected is {mape_corrected:0.2f}")
mape_array.append([mape, mape_corrected])
plt.figure(figsize=(20, 10))
plt.title('Humidity')
plt.xlabel('Time')
plt.ylabel('Humidity')
plt.plot(humidity.index, humidity['humidity_openweather'], label='Accuweather')
plt.plot(humidity.index, humidity['humidity_openweather_corrected'], label='Openweather corrected')
plt.plot(humidity.index, humidity['hummidity_addvantage'], label='Addvantage')
plt.legend()
plt.tight_layout()
plt.savefig('Z:\Projects\ΑΥΓΕΙΑΣ\ΕΓΓΡΑΦΑ/humidity_openweather_addvantage.png', dpi=300)
plt.show()

wind_speed = data[['wind_speed_openweather', 'wind_speed_openweather_corrected', 'wind_speed_addvantage']]
mape = mean_absolute_error(wind_speed['wind_speed_openweather'], wind_speed['wind_speed_addvantage'])*100
print(f"MAPE for wind speed is {mape:.2f}")
mape_corrected = mean_absolute_error(wind_speed['wind_speed_openweather_corrected'], wind_speed['wind_speed_addvantage'])*100
print(f"MAPE for wind speed corrected is {mape_corrected:0.2f}")
mape_array.append([mape, mape_corrected])
plt.figure(figsize=(20, 10))
plt.title('Wind speed')
plt.xlabel('Time')
plt.ylabel('Wind speed')
plt.plot(wind_speed.index, wind_speed['wind_speed_openweather'], label='Accuweather')
plt.plot(wind_speed.index, wind_speed['wind_speed_openweather_corrected'], label='Openweather corrected')
plt.plot(wind_speed.index, wind_speed['wind_speed_addvantage'], label='Addvantage')
plt.legend()
plt.tight_layout()
plt.savefig('Z:\Projects\ΑΥΓΕΙΑΣ\ΕΓΓΡΑΦΑ/wind_speed_openweather_addvantage.png', dpi=300)
plt.show()



mape_df = pd.DataFrame(mape_array)


mape_df.index = ['Temperature', 'Humidity', 'Wind speed']

mape_df.columns = ['MAPE', 'MAPE corrected']

mape_df.plot.bar(figsize=(20, 10), rot=0, )

plt.ylabel('MAPE')
plt.legend()
plt.tight_layout()
plt.savefig('Z:\Projects\ΑΥΓΕΙΑΣ\ΕΓΓΡΑΦΑ\mape_openweather_addvantage.png', dpi=300)
plt.show()


mape_df['Percentage decrease'] = (mape_df['MAPE'] - mape_df['MAPE corrected'])/mape_df['MAPE']*100
# mape_df['Percentage decrease'] = mape_df['Percentage decrease'].apply(lambda x: f"{x:0.2f}%")
mape_df.drop('MAPE', inplace=True, axis=1)
mape_df.drop('MAPE corrected', inplace=True, axis=1)
print(mape_df)
ax = mape_df.plot.bar(figsize=(20, 10), rot=0 )
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f%%', fontsize=14)
plt.title('Percentage decrease in MAPE')
plt.ylabel('Percentage decrease')
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.plot()
plt.savefig('Z:\Projects\ΑΥΓΕΙΑΣ\ΕΓΓΡΑΦΑ\percentage_decrease_openweather_addvantage.png', dpi=300)
plt.show()


