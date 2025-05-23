import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from haversine import haversine, Unit
import numpy as np

def load_data(data):
    df = pd.read_csv(data)
    return df

class Pipeline:
    def __init__(self, data, time_window_minutes=30):# Ændrer 30, for at ændre standarden. Alternativt ændrer i main()
        self.data = data
        self.time_window_minutes = time_window_minutes

    def drop_columns(self):
        columns_to_drop = ['Navigational status', 'ROT', 'Heading', 'Data source type', 'Ship type']
        self.data = self.data.drop(columns=[col for col in columns_to_drop if col in self.data.columns], errors='ignore')
        return self

    def drop_duplicates(self):
        self.data = self.data.drop_duplicates()
        return self
    
    def filter_time_mmsi(self):
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
        self.data['Window'] = self.data['Timestamp'].dt.floor(f'{self.time_window_minutes}min')
        return self

    def handle_missing_values(self):
        """Håndterer NaN-værdier i SOG (interpolation) og COG (forward fill) per MMSI."""
        
        def interpolate_nan_blocks(data, column):
            data = data.copy().sort_values("Timestamp").reset_index(drop=True)
            data[column] = data[column].interpolate(method='linear', limit_direction='both', limit_area='inside')
            return data

        def fill_column_ffill_bfill(data, column):
            data = data.copy().sort_values("Timestamp").reset_index(drop=True)
            data[column] = data[column].ffill().bfill()
            return data

        def process_column_by_mmsi(data, column, method):
            mmsi_list = data[data[column].isna()]['MMSI'].unique().tolist()
            filled_data = []

            for mmsi in mmsi_list:
                mmsi_data = data[data['MMSI'] == mmsi]
                mmsi_data = method(mmsi_data, column)
                filled_data.append(mmsi_data)

            untouched_data = data[~data['MMSI'].isin(mmsi_list)]
            return pd.concat([untouched_data, *filled_data]).sort_values("Timestamp").reset_index(drop=True)

        self.data = process_column_by_mmsi(self.data, 'SOG', interpolate_nan_blocks)

        self.data = process_column_by_mmsi(self.data, 'COG', fill_column_ffill_bfill)

        return self

    def filter_windows_with_minimum_pings(self, min_count=3):
        """
        Fjerner alle (MMSI, Window) kombinationer med færre end `min_count` målinger.
        Det sikrer, at feature engineering-beregninger bliver lavet, da de fleste kræver mindst 2 værdier(sættes til 3 grundet std_acceleration).
        """
        count_df = self.data.groupby(['MMSI', 'Window'])['SOG'].count().reset_index(name='count')
        valid_windows = count_df[count_df['count'] >= min_count][['MMSI', 'Window']]
        self.data = pd.merge(self.data, valid_windows, on=['MMSI', 'Window'], how='inner')
        return self


    def feature_eng_SOG_mean_std(self):
        grouped = self.data.groupby(['MMSI', 'Window'])['SOG'].agg(['mean', 'std']).reset_index()
        grouped.rename(columns={'mean': 'Average_SOG', 'std': 'STD_SOG'}, inplace=True)
        self.data = pd.merge(self.data, grouped, on=['MMSI', 'Window'], how='left')
        return self
    
    def calculate_delta_sog(self):
        delta_sog_df = (self.data
                        .groupby(['MMSI', 'Window'])['SOG']
                        .agg(max_SOG='max', min_SOG='min')
                        .reset_index())
        delta_sog_df['delta_SOG'] = delta_sog_df['max_SOG'] - delta_sog_df['min_SOG']
        self.data['MMSI'] = self.data['MMSI'].astype(str)
        delta_sog_df['MMSI'] = delta_sog_df['MMSI'].astype(str)
        self.data = pd.merge(self.data, delta_sog_df, on=['MMSI', 'Window'], how='left')
        return self

    def feature_eng_acceleration(self):
        self.data.sort_values(['MMSI','Timestamp'], inplace=True)
        self.data['sog_diff'] = self.data.groupby('MMSI')['SOG'].diff()
        self.data['dt_s'] = self.data.groupby('MMSI')['Timestamp'].diff().dt.total_seconds()
        
        max_interval_s = self.time_window_minutes * 60
        acc = self.data['sog_diff'] / self.data['dt_s']
        acc[self.data['dt_s'] > max_interval_s] = np.nan
        
        acc.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data['acceleration'] = acc
        self.data['abs_acceleration'] = acc.abs()
        return self

    def feature_eng_turning_intensity(self):
        # Compute bearing and turning rate
        df = self.data.sort_values(['MMSI','Timestamp']).copy()
        df['prev_lat'] = df.groupby('MMSI')['Latitude'].shift(1)
        df['prev_lon'] = df.groupby('MMSI')['Longitude'].shift(1)
        prev_lat = np.radians(df['prev_lat']); prev_lon = np.radians(df['prev_lon'])
        curr_lat = np.radians(df['Latitude']); curr_lon = np.radians(df['Longitude'])
        dlon = curr_lon - prev_lon
        x = np.sin(dlon)*np.cos(curr_lat)
        y = np.cos(prev_lat)*np.sin(curr_lat) - np.sin(prev_lat)*np.cos(curr_lat)*np.cos(dlon)
        df['bearing'] = (np.degrees(np.arctan2(x,y)) + 360) % 360
        df['angle_diff'] = df.groupby('MMSI')['bearing'].diff()
        df['angle_diff'] = (df['angle_diff'] + 180) % 360 - 180
        df['time_diff_heading'] = df.groupby('MMSI')['Timestamp'].diff().dt.total_seconds()
        df['turning_intensity'] = (df['angle_diff'].abs() / df['time_diff_heading']).replace([np.inf,-np.inf], np.nan)
        # Spike detection (95th percentile)
        thresh = df['turning_intensity'].quantile(0.95)
        df['prev_turn'] = df.groupby('MMSI')['turning_intensity'].shift(1)
        df['next_turn'] = df.groupby('MMSI')['turning_intensity'].shift(-1)
        df['spike'] = ((df['turning_intensity']>df['prev_turn']) & (df['turning_intensity']>df['next_turn']) & (df['turning_intensity']>thresh))
        self.data = df.drop(columns=['prev_lat','prev_lon','bearing','angle_diff','time_diff_heading','prev_turn','next_turn'])
        return self
            

    def calculate_distance_to_coast(self, coastline_path=r"C:\Program Files (x86)\2 Semester python work\Projekt\P2_DAKI2_Hold_4\MAIN_Workspace\program\coastline_folder\ne_10m_coastline.shp"):
        self.data['geometry'] = self.data.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
        gdf = gpd.GeoDataFrame(self.data, geometry='geometry', crs="EPSG:4326")
        coastline = gpd.read_file(coastline_path)
        bbox = box(7.5, 54.5, 13.0, 58.0)
        denmark_bounds = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
        coastline = gpd.overlay(coastline, denmark_bounds, how='intersection')
        gdf_proj = gdf.to_crs(epsg=25832)
        coastline_proj = coastline.to_crs(epsg=25832)
        gdf_proj['distance_to_coast_m'] = gdf_proj.geometry.apply(
            lambda point: coastline_proj.distance(point).min()
        )
        self.data['distance_to_coast_m'] = gdf_proj['distance_to_coast_m']
        return self

    def feature_eng_close_to_coast(self, threshold=500):
        self.data['close_to_coast'] = (self.data['distance_to_coast_m'] >= threshold).astype(int)
        return self

    def calculate_density_and_avg_distance(self):
        densities = []
        for mmsi, group in self.data.groupby(['MMSI', 'Window']):
            coords = list(zip(group['Latitude'], group['Longitude']))
            n = len(coords)

            if n < 2:
                avg_distance_between_pings = 0.0
                density = 0.0
            else:
                # Kun sekventielle afstande (i til i+1)
                distances = [haversine(coords[i], coords[i+1], unit=Unit.METERS)
                            for i in range(n - 1)]
                avg_distance_between_pings = np.mean(distances) if distances else 0.0
                density = 1 / (avg_distance_between_pings + 1e-5)

            group = group.copy()
            group['density'] = density
            group['avg_distance_between_pings'] = avg_distance_between_pings
            densities.append(group)

        self.data = pd.concat(densities)
        return self



    def aggregate_30min_intervals(self):
        def majority_label(labels):
            mode = labels.mode()
            return mode.iloc[0] if not mode.empty else np.nan

        agg = (
            self.data
            .groupby(['MMSI','Window'])
            .agg(
                Average_SOG=('Average_SOG', 'mean'),
                STD_SOG=('STD_SOG', 'mean'),
                distance_to_coast_m=('distance_to_coast_m', 'mean'),
                far_from_coast=('close_to_coast', lambda x: round(x.mean())),
                mean_acceleration=('acceleration', 'mean'),
                std_acceleration=('acceleration', 'std'),
                mean_abs_acceleration=('abs_acceleration', 'mean'),
                mean_turning_intensity = ('turning_intensity','mean'),
                max_turning_intensity = ('turning_intensity','max'),
                turning_spike_count = ('spike', 'sum'),                
                avg_distance_between_pings=('avg_distance_between_pings', 'mean'),
                density=('density', 'mean'),
                max_SOG=('max_SOG', 'max'),
                min_SOG=('min_SOG', 'min'),
                delta_SOG=('delta_SOG', 'max'),
                Behavior_Label=('Behavior Label', majority_label),
            )
            .reset_index()
        )
        agg['far_from_coast'] = agg['far_from_coast'].astype(int)

        label_map = {'stille': 0, 'transport': 1, 'fiskeri': 2}
        agg['Behavior_Label'] = agg['Behavior_Label'].map(label_map)

        self.data = agg
        return self


    def retrieve_data(self):
        self.data.set_index('Window', inplace=True)
        return self.data

def main():
    data = r'C:\Program Files (x86)\2 Semester python work\Projekt\P2_DAKI2_Hold_4\MAIN_Workspace\program\Data_sæts\labeled_ships.csv'
    time_window_minutes = 40 # Ændrer for at ændre tidsintervallet i koden
    df = load_data(data)

    # Hvis du kun vil teste en enkelt MMSI så kør de næste 2 linjer
    #unique_mmsi = df['MMSI'].unique()[:1]  # Use a small subset for testing
    #df_subset = df[df['MMSI'].isin(unique_mmsi)]

    # Hvis du vil køre hele datasættet kør næste linje og slet de 2 linjer ovenfor
    df_subset = df  # Brug hele datasættet


    pipeline = Pipeline(df_subset, time_window_minutes=time_window_minutes) # For at ændre tidsintervallet kik i toppen af main()
    processed_data = (pipeline
                      .drop_columns()
                      .drop_duplicates()
                      .filter_time_mmsi()
                      .handle_missing_values()
                      .filter_windows_with_minimum_pings()
                      .feature_eng_SOG_mean_std()
                      .calculate_delta_sog()
                      .calculate_distance_to_coast()
                      .feature_eng_close_to_coast()
                      .calculate_density_and_avg_distance()
                      .feature_eng_acceleration()
                      .feature_eng_turning_intensity()
                      .aggregate_30min_intervals()
                      .retrieve_data())
    
    # Gemmer som en Excel-fil
    #processed_data.to_excel("processed_data_one_ship.xlsx", index=False)

    # Gemmer som CSV-fil
    processed_data.to_csv(f"processed_data_{time_window_minutes}min.csv")



    print("Processed data head:\n", processed_data.head(10))
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Number of unique windows: {processed_data.index.nunique()}")


if __name__ == "__main__":
    main()


# Før pipeline køres, så sikrer at dataframe vejen er korrekt. Findes i funktionen main()
# Sikrer også at funktionen def calculate_distance_to_coast(self, coastline_path=r"ne_10m_coastline.shp") har stien til "ne_10m_coastline.shp" korrekt