# Analysis Data
Processed data contains four files; delay_train_inputs.csv, delay_test_inputs.csv, GSE_train_inputs.csv and GSE_test_inputs.csv. They contain train and test data for delay count prediction and gated station entry prediction, respectively. All files contain each day where data is available for the target metric and all additional metrics. For delay inputs, the full date range is from January 1, 2019 to July 29, 2023 (1671 days); for gated station entry inputs, the full date range is from January 1, 2014 to June 30, 2025 (4199 days). Each task is split chronologically into 80% train rows and 20% test rows before scaling. Scaled columns in the test files are transformed using the scaler fit on the corresponding train file.
Columns contain, for all four files:
* Column 1: Date, in format YYYY-MM-DD; labeled ```Date```.
* Column 2: The metric to predict, labeled ```Total_Delays``` or ```Gated_Station_Entries``` depending on file.
* Column 3: Day of week, labeled ```Day_of_Week```. Monday is 0, Tuesday is 1, ... Sunday is 6.
* Column 4: Season, labeled ```Season```. Winter is 0, Spring is 1, Summer is 2 and Autumn is 3.
* Column 5: Pressure in hectopascals, labeled ```Pressure```.
* Column 6: Wind speed in km/h, labeled ```Wind_Speed```.
* Column 7: Average temperature in Celsius, labeled ```Average_Temperature```.
* Column 8: Average precipitation in millilitres, labeled ```Precipitation```.
* Columns 9 to 15: A one-hot encoding of day of week, labeled (in order) ```Is_Monday, Is_Tuesday, ... Is_Sunday```.
* Columns 16 to 19: A one-hot encoding of season, labeled (in order) ```Is_Winter, Is_Spring, Is_Summer, Is_Autumn```.
* Columns 20 to 36: Scaled versions of values in columns 3 to 19, scaled with scikit-learn's StandardScaler. Prefixed with ```S_```, e.g. ```S_Day_Of_Week```.
