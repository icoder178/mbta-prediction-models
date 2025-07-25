# Input Data
Raw input data for analysis. Directly cloning yields the files "GSE_by_year.zip", "MBTA_Service_Alerts.csv.zip", and "boston_weather_data.csv". Each are explained below.
## GSE_by_year.zip
Gated station entries by year. As this contains over 10 years of data and has not been aggregated, it is compressed to avoid issues with Github. Decompressing yields a folder "GSE_by_year" containing files "GSE_2014.csv", "GSE_2015.csv", ... "GSE_2025.csv" – raw gated station entries by year. This folder name is in the .gitignore to prevent accidental uploading to the repo.
Within each CSV file, there are 6 columns:
* Column 1: ```service_date``` – the date of the current data point.
* Column 2 to 5 (__IGNORED__): ```time_period,stop_id,station_name,route_or_line``` – various locational attributes of the data point. As we are aggregating by day, these are ignored.
* Column 6: ```gated_entries``` – the gated station entries for that data point. Sometimes contains non-integers due to splitting of entries between lines (e.g. 20% of station entries are considered going to the Blue Line, and 80% to the Green Line); however, total entries by day are still integers.
## MBTA_Service_Alerts.csv.zip
MBTA service alerts. As this has not been aggregated, it is over 140 MB uncompressed and so is compressed to avoid issues with Github. Decompressing yields a file "MBTA_Service_Alerts.csv.zip", which is in the .gitignore to prevent accidental uploads. Within this CSV file, there are 24 columns:
* Columns 1 to 3 (__IGNORED__): ```alert_id,gui_mode_name,alert_time_type``` – Alert details. As we are totalling delays by day, these are ignored.
* Column 4: ```effect_name``` – if "Delay", this is considered one set of delays.
* Columns 5 to 13 (__IGNORED__): ```effect_code,cause_name,cause_code,affent_list,severity_name,severity_code,header,details,url``` – details on cause, effect, and severity. As we are totalling delays by day, these are ignored.
* Columns 14 to 15: ```notif_start,notif_end``` – when the notification lasted from. Any day in this range is considered a day with one "delay"; for example, if the notification started 4PM 2025/6/20 and ended 2AM 2025/6/21, we add one delay to both 2025/6/20 and 2025/6/21.
* Columns 16 to 24 (__IGNORED__): ```created_dt,last_modified_dt,closed_dt,alert_lifecycle,color,service_effect_text,short_header,timeframe_text,ObjectId``` – various technical details. As we are totalling delays by day, these are ignored.