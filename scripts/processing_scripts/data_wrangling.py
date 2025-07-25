import pandas as pd
import numpy as np
import copy
import sys
from sklearn.preprocessing import StandardScaler

# so that appending "no_debug" when running program skips printing any debug messages
def print_debug(val):
    if 'no_debug' in sys.argv:
        return
    print(val)

# computes day of week; Monday is 0, Tuesday is 1, ... Sunday is 6
def compute_day_of_week(data):
    _data = pd.to_datetime(data)
    return _data.dt.dayofweek

# computes season; Winter is 0, Spring is 1, Summer is 2, Autumn is 3
def compute_season(data):
    _data = pd.to_datetime(data)
    season = []
    for current_datetime in _data:
        current_month = current_datetime.month
        if current_month >= 12 or current_month <= 2: season.append(0)
        elif current_month >= 3 and current_month <= 5: season.append(1)
        elif current_month >= 6 and current_month <= 8: season.append(2)
        else: season.append(3)
    return pd.Series(season)

# computes one-hot encoding of day
def compute_day_ohe(data):
    one_hot_encoding = [(data == x) for x in range(0,7)]
    one_hot_encoding = [list(row) for row in zip(*one_hot_encoding)]
    one_hot_encoding = pd.DataFrame(one_hot_encoding,columns = ['Is_Monday','Is_Tuesday','Is_Wednesday','Is_Thursday','Is_Friday','Is_Saturday','Is_Sunday'])
    return one_hot_encoding

# computes one-hot encoding of season
def compute_season_ohe(data):
    one_hot_encoding = [(data == x) for x in range(0,4)]
    one_hot_encoding = [list(row) for row in zip(*one_hot_encoding)]
    one_hot_encoding = pd.DataFrame(one_hot_encoding,columns = ['Is_Winter','Is_Spring','Is_Summer','Is_Autumn'])
    return one_hot_encoding

# loads GSE data and does aggregation and rearrangement
def gse_data():
    gse_file_names = [f"../../data/input_data/GSE_by_year/GSE_{year}.csv" for year in range(2014,2026)]
    totals_by_day = pd.DataFrame()
    for file_name in gse_file_names:
        current_file = pd.read_csv(file_name)
        print_debug(f"Currently beginning file \"{file_name}\", file length {len(current_file)}")
        agg_file = current_file.groupby('service_date',as_index=False)['gated_entries'].sum()
        totals_by_day = pd.concat([totals_by_day,agg_file])
    totals_by_day.columns = ['Date','Gated_Station_Entries']
    totals_by_day = totals_by_day.set_index('Date')
    totals_by_day = totals_by_day.sort_index()
    totals_by_day['Gated_Station_Entries'] = totals_by_day['Gated_Station_Entries'].round().astype(int)
    return totals_by_day

# loads weather data and rearranges to necessary format
def weather_data():
    weather_file = pd.read_csv("../../data/input_data/boston_weather_data.csv")
    needed_names = ['time','pres','wspd','tavg','prcp']
    to_concat = [weather_file[name] for name in needed_names]
    clipped_weather_file = pd.concat(to_concat,axis=1)
    clipped_weather_file.columns = ['Date','Pressure','Wind_Speed','Average_Temperature','Precipitation']
    clipped_weather_file = clipped_weather_file.set_index('Date')
    clipped_weather_file = clipped_weather_file.sort_index()
    return clipped_weather_file

# loads delay data and does aggregation and rearrangement
def delay_data():
    service_alerts = pd.read_csv("../../data/input_data/MBTA_Service_Alerts.csv")
    service_alerts['notif_start'] = service_alerts['notif_start'].str[:10]
    service_alerts['notif_end'] = service_alerts['notif_end'].str[:10]
    service_alerts['notif_start'] = pd.to_datetime(service_alerts['notif_start'],format="%Y/%m/%d",errors='coerce')
    service_alerts['notif_end'] = pd.to_datetime(service_alerts['notif_end'],format="%Y/%m/%d",errors='coerce')
    service_alerts = service_alerts.dropna(subset=['notif_start','notif_end'])
    cycle_cnt = 0
    display_when = 10000
    totaled_delays = pd.DataFrame()
    totaled_delays['Total_Delays'] = []
    days = {}
    for i in range(len(service_alerts['notif_start'])):
        if service_alerts['effect_name'].iloc[i] == 'Delay':
            current_datetime = service_alerts['notif_start'].iloc[i]
            end_datetime = service_alerts['notif_end'].iloc[i]
            while current_datetime <= end_datetime:
                datetime_str = current_datetime.strftime("%Y-%m-%d")
                if datetime_str not in days:
                    days[datetime_str] = True
                    new_row = pd.DataFrame([{'Total_Delays': 0}], index=[datetime_str])
                    totaled_delays = pd.concat([totaled_delays,new_row])
                totaled_delays.loc[datetime_str,'Total_Delays'] += 1
                current_datetime += pd.DateOffset(days=1)
        cycle_cnt += 1
        if cycle_cnt%display_when == 0:
            print_debug(f"{cycle_cnt} cycles done.")
    totaled_delays = totaled_delays.sort_index()
    totaled_delays['Total_Delays'] = totaled_delays['Total_Delays'].round().astype(int)
    return totaled_delays

# utility function for merging dataframes by index
def merge_func(arr):
    return_value = arr[0]
    for i in range(1,len(arr)):
        return_value = pd.merge(return_value, arr[i], left_index=True, right_index=True, how='outer')
    return return_value.dropna()

# collects all additional data and merges it together
def additional_data():
    weather = weather_data()
    date_list = copy.deepcopy(pd.DataFrame(weather.index)['Date'])
    weather = weather.reset_index()
    day_of_week = pd.DataFrame(compute_day_of_week(date_list))
    day_of_week.columns = ['Day_of_Week']
    season = pd.DataFrame(compute_season(date_list))
    season.columns = ['Season']
    day_of_week_ohe = compute_day_ohe(day_of_week['Day_of_Week'])
    season_ohe = compute_season_ohe(season['Season'])
    return_value = merge_func([day_of_week,season,weather,day_of_week_ohe,season_ohe])
    return return_value.set_index('Date')

# scales data recieved
def scaled_data(data):
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(data),columns="S_"+data.columns)
    scaled.index = data.index
    return scaled

# cheatsheet for easier writing of data positions in neural net training
def output_cheatsheet(data):
    if "no_cheatsheet" in sys.argv:
        return
    for i in range(len(data.columns)):
        print(f"{i+1}: {data.columns[i]}")
    print("")

# gets data, processes and outputs for gated station entries and delay counts
def main():
    print_debug("Gated Station Entries:")
    gse = gse_data()
    print_debug("Delay:")
    delay = delay_data()
    print_debug("Additional Data:")
    additional = additional_data()
    scaled = scaled_data(additional)

    gse_result = merge_func([gse,additional,scaled])
    gse_result.to_csv("../../data/analysis_data/GSE_inputs.csv")
    output_cheatsheet(gse_result)
    delay_result = merge_func([delay,additional,scaled])
    delay_result.to_csv("../../data/analysis_data/delay_inputs.csv")
    output_cheatsheet(delay_result)

main()