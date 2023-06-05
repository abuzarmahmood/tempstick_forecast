import pandas as pd
import pylab as plt
import numpy as np
import sys
sys.path.append('/media/bigdata/projects/tempstick_forecast/src')
from load_data import return_dat
from xgboost import XGBRegressor, plot_importance
import shap
from sklearn.preprocessing import StandardScaler
import seaborn as sns

dat_dict = return_dat()
sensor_dat = dat_dict['sensor_dat']
forecast_dat = dat_dict['forecast_dat']

# Resample to hourly
sensor_dat = sensor_dat.resample('H').mean()
forecast_dat = forecast_dat.resample('H').mean()

dat_names = ['sensor_dat', 'forecast_dat']
fig, ax = plt.subplots(2,1,sharex=True)
for i, (this_dat, this_ax) in enumerate(zip([sensor_dat, forecast_dat], ax)):
    for this_vals in this_dat.columns:
        this_ax.plot(this_dat.index, this_dat[this_vals], label=this_vals,
                     marker='o', linestyle='none')
    this_ax.legend(loc='upper left')
    this_ax.set_ylabel('Temperature (F)')
    this_ax.set_xlabel('Date')
    this_ax.set_title(dat_names[i])
    plt.tight_layout()
plt.show()

# Plot actual dat vs forecast dat for each feature
merged_dat = sensor_dat.merge(forecast_dat, left_index=True, right_index=True,
                              suffixes=('_actual', '_forecast'))
merged_dat = merged_dat.dropna()
# Generate scalers to invert transform later
temp_scaler = StandardScaler().fit(merged_dat['temp_actual'].values.reshape(-1,1))
humidity_scaler = StandardScaler().fit(merged_dat['humidity_actual'].values.reshape(-1,1))

# Standardize
scaler = StandardScaler()
merged_dat = pd.DataFrame(scaler.fit_transform(merged_dat),
                          merged_dat.index, merged_dat.columns)

merged_dat.plot()
plt.show()

fig, ax = plt.subplots(2,1)
ax[0].plot(merged_dat.temp_actual, merged_dat.temp_forecast, 
           marker='o', linestyle='none', alpha=0.5)
ax[0].set_xlabel('Actual Temperature (F)')
ax[0].set_ylabel('Forecast Temperature (F)')
ax[0].set_title('Temperature')
ax[1].plot(merged_dat.humidity_actual, merged_dat.humidity_forecast, 
           marker='o', linestyle='none', alpha=0.5)
ax[1].set_xlabel('Actual Humidity (%)')
ax[1].set_ylabel('Forecast Humidity (%)')
ax[1].set_title('Humidity')
plt.tight_layout()
plt.show()


############################################################
############################################################

# Features for sensor data prediction
# 1. Time of day
# 2. Sensor history
# 3. Weather history
# 4. Weather forecast (future)

# Model to use:
# XGBoost

# Use SHAP analysis to determine feature importance

# Predictive window:
# 1 week

############################################################

# Prepare features for sensor data prediction

# Define parameters
max_lag = 24*7 
pred_window = 24*7

# Generate features
start_time = merged_dat.index[0] + pd.Timedelta(hours=max_lag)
max_time = merged_dat.index[-1] - pd.Timedelta(hours=pred_window)

y_cols = ['temp_actual', 'humidity_actual']
y_dat = merged_dat[y_cols]
# y --> snippets of sensor data which are pred_window long
y_inds = y_dat[start_time:max_time].index
y_list = []
for this_time in y_inds:
    this_y = y_dat.loc[this_time:this_time+pd.Timedelta(hours=pred_window-1)]
    this_y = this_y.values.flatten(order='F')
    y_list.append(this_y)
y_names = ['{}_{}'.format(i, j) for i in y_cols for j in ['t+{}'.format(k) for k in range(pred_window)]]
y = pd.DataFrame(y_list, index=y_inds, columns=y_names)

#y.iloc[0].plot()
#plt.show()

history_X_cols = ['temp_actual', 'humidity_actual', 'temp_forecast', 'humidity_forecast']
future_X_cols = ['temp_forecast', 'humidity_forecast']

feature_list = []
for i in y.index:
        wanted_history = merged_dat.loc[i-pd.Timedelta(hours=max_lag):i]
        wanted_future = merged_dat.loc[i:i+pd.Timedelta(hours=pred_window)]
        wanted_history = wanted_history[history_X_cols]
        wanted_future = wanted_future[future_X_cols]
        hour = i.hour
        flat_features = np.concatenate(
                [wanted_history.values.flatten(order='F'), 
                 wanted_future.values.flatten(order='F'), [hour]]) 
        feature_list.append(flat_features)
wanted_history_names = ['{}_{}'.format(i, j) for i in history_X_cols \
        for j in ['t-{}'.format(k) for k in range(len(wanted_history))]]
wanted_future_names = ['{}_{}'.format(i, j) for i in future_X_cols \
        for j in ['t+{}'.format(k) for k in range(len(wanted_future))]]
feature_names = wanted_history_names + wanted_future_names + ['hour']
X = pd.DataFrame(feature_list, index=y.index, columns=feature_names)

# Split into train and test
train_frac = 0.8
train_size = int(X.shape[0]*train_frac)
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

# Train model
model = XGBRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)

# Plot predictions
vmin, vmax = np.min([y_test.values, y_pred.values]), np.max([y_test.values, y_pred.values])
fig, ax = plt.subplots(2,1,sharex=True, sharey=True)
ax[0].imshow(y_test.T, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)
ax[0].set_title('Actual')
ax[1].imshow(y_pred.T, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)
ax[1].set_title('Predicted')
plt.show()

# Plot example traces
# Plot temp and humidity separately
delta_t = [x.split('+')[1] for x in y_names if 'temp' in x]
feature_name = [x.split('_')[0] for x in y_names]

ind = 0
fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(delta_t, y_test.iloc[ind, [i for i,x in enumerate(feature_name) if x=='temp']], 
           marker='o', linestyle='-', label='Actual')
ax[0].plot(delta_t, y_pred.iloc[ind, [i for i,x in enumerate(feature_name) if x=='temp']],
           marker='o', linestyle='-', label='Predicted')
ax[0].set_ylabel('Temperature (F)')
ax[0].legend()
ax[1].plot(delta_t, y_test.iloc[ind, [i for i,x in enumerate(feature_name) if x=='humidity']],
           marker='o', linestyle='-', label='Actual')
ax[1].plot(delta_t, y_pred.iloc[ind, [i for i,x in enumerate(feature_name) if x=='humidity']],
           marker='o', linestyle='-', label='Predicted')
ax[1].set_ylabel('Humidity (%)')
ax[1].set_xlabel('Time into future (hours)')
ax[1].legend()
plt.show()

# Plot feature importance
fig, ax = plt.subplots()
plot_importance(model, ax=ax)
# Change y axis labels to feature names
ax.set_yticklabels(feature_names)
# adjust subplots to make room for the feature names
plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)
plt.show()

# Calculate prediction error for time into future

# Convert y_test and y_pred into long format

y_test_long = y_test.stack().reset_index()
y_test_long.columns = ['time', 'feature', 'value']
y_test_long['feature_name'] = [x.split('_')[0] for x in y_test_long.feature]
y_test_long['delta_time'] = [int(x.split('+')[1]) for x in y_test_long.feature]

y_pred_long = y_pred.stack().reset_index()
y_pred_long.columns = ['time', 'feature', 'value']
y_pred_long['feature_name'] = [x.split('_')[0] for x in y_pred_long.feature]
y_pred_long['delta_time'] = [int(x.split('+')[1]) for x in y_pred_long.feature]

# convert feature_name into columns
y_test_long = y_test_long.pivot_table(index=['time', 'delta_time'],
                                      values='value', columns='feature_name')
y_pred_long = y_pred_long.pivot_table(index=['time', 'delta_time'],
                                      values='value', columns='feature_name')
merged_y = pd.merge(y_test_long, y_pred_long, on=['time', 'delta_time'],
                    suffixes=['_actual', '_pred'])

# Unscale data
temp_cols = [x for x in merged_y.columns if 'temp' in x]
humidity_cols = [x for x in merged_y.columns if 'humidity' in x]
for this_col in temp_cols:
    merged_y[this_col] = temp_scaler.inverse_transform(merged_y[this_col].values.reshape(-1,1))
for this_col in humidity_cols:
    merged_y[this_col] = humidity_scaler.inverse_transform(merged_y[this_col].values.reshape(-1,1))

# Plot scaled actual vs prediction
ind = 30
wanted_ind = y_pred.index[ind]
wanted_dat = merged_y.loc[(wanted_ind, slice(None)), :]
wanted_dat = wanted_dat.reset_index().drop('time', axis=1)
wanted_dat = wanted_dat.set_index('delta_time')
wanted_dat = wanted_dat.sort_index()
wanted_dat.plot(marker='o', linestyle='-', figsize=(10,5))
plt.xlabel('Time into future (hours)')
plt.show()

# Convert to long format again with feature and [actual, pred] columns
long_merged_y = merged_y.stack().reset_index()
long_merged_y.columns = ['time', 'delta_time', 'feature_name', 'value']
long_merged_y['feature'] = [x.split('_')[0] for x in merged_y.feature_name]
long_merged_y['data_type'] = [x.split('_')[1] for x in merged_y.feature_name]

# Calculate error
feature_types = ['temp', 'humidity']
data_types = ['actual', 'pred']
error_frame = pd.DataFrame()
for this_feature in feature_types:
    feature_dat = long_merged_y[merged_y.feature==this_feature]
    feature_dat.set_index(['time', 'delta_time'], inplace=True)
    actual_dat = feature_dat[feature_dat.data_type=='actual']
    pred_dat = feature_dat[feature_dat.data_type=='pred']
    error = actual_dat.value - pred_dat.value
    error = pd.DataFrame(error)
    error['feature'] = this_feature
    error_frame = pd.concat([error_frame, error], axis=0)

error_frame['pred_error'] = error_frame['pred_error'].abs()

error_frame.rename(columns={'value': 'pred_error', 'feature':'feature_name'}, inplace=True)
error_frame.reset_index(inplace=True)

# Plot error and overlay median error
sns.catplot(x='delta_time', y='pred_error', row='feature_name', data=error_frame, kind='point')
plt.show()
