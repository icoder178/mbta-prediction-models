Incomplete README file.
# mbta-prediction-models
Testing the effectiveness of various models to predict from MBTA data.
### Data Sources
* Gated Station Entries: https://mbta-massdot.opendata.arcgis.com/datasets/7859894afb5641ce91a2bb03599fdf5b/about
* MBTA Service Alerts: https://gis.data.mass.gov/datasets/MassDOT::mbta-service-alerts/about
* Boston Weather Data: Meteostat API, run with code/meteostat_import.py (pip install meteostat needed)
### Necessary Data
* Date (0, index, in order)
* Total_Delays or Gated_Station_Entries (1)
* Day_of_Week (2)
* Season (3)
* Pressure (4)
* Wind_Speed (5)
* Average_Temperature (6)
* Precipitation (7)
* Is_Monday ... Is_Sunday (8 ... 14)
* Is_Winter ... Is_Autumn (15 ... 18)
* S_Day_of_Week ... S_Is_Autumn (19 ... 35)

### To Test
Data from 5 previous days to be used. 5 each, 2 prediction targets for a total of 2*5*12 = 120 experiments.
* No Additional Data
* Additional Weather, Day of Week, Season
* Additional Normalized Weather, Day of Week, Season
* Additional Weather, Day of Week OHE, Season OHE
* Additional Normalized Weather, Day of Week OHE, Season OHE

### To Display
Bar graph, of 12 models. For each, RMSE for:
* 5 Previous Days, No Additional Data
* 5 Previous Days, Weather, Day of Week, Season (max. of all models with additional data)
