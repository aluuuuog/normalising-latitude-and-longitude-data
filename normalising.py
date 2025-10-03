# My First Python Program for Processing Geographical Data!
# I'm using code I found online (and in my notes) to try out
# three different ways to "normalize" or "fix" latitude and longitude
# so a computer can use them better for machine learning.

# --- 1. Setting up my tools (Importing Libraries) ---

# pandas is for making and managing my tables (DataFrames).
import pandas as pd
# numpy is great for all sorts of math, especially the sin/cos stuff.
import numpy as np
# geopandas is a special tool for map stuff, like changing coordinate systems.
import geopandas
# Point is a helper from shapely to turn coordinates into map points.
from shapely.geometry import Point
# MinMaxScaler is the part that squishes numbers down to be between 0 and 1.
from sklearn.preprocessing import MinMaxScaler
# I need these specific math functions for the distance formula (Haversine).
from math import radians, sin, cos, sqrt, asin


print("--- Starting my project to transform location numbers! ---")
print("----------------------------------------------------------\n")


# --- 2. My Sample Input Data ---
# I'm making up three simple taxi trips, all starting in Manhattan and ending in Brooklyn.
# Latitude and Longitude are in the standard WGS 84 system.
print("Step 2: Creating a small table of raw trip data (in Degrees)...")
trip_data = {
    'trip_id': [1, 2, 3],
    'pickup_lat': [40.7580, 40.7585, 40.7600],
    'pickup_lon': [-73.9855, -73.9860, -73.9840],
    'dropoff_lat': [40.6900, 40.6905, 40.6920],
    'dropoff_lon': [-73.9700, -73.9705, -73.9680]
}
my_data_table = pd.DataFrame(trip_data)
print("Initial Data Table:")
print(my_data_table[['trip_id', 'pickup_lat', 'pickup_lon']].head())
print("\n")


# --- 3. Normalization Method 1: Calculating Haversine Distance ---
# This is called "Derived Normalization." I turn four coordinates into one meaningful number: distance!

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    The Haversine formula finds the shortest distance between two points
    on a sphere (the Earth). The result is in kilometers (km).
    """
    Earth_Radius_km = 6371 # The average radius of Earth in km
    
    # First, convert all the degrees into "radians" for the math functions
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    
    # Calculate the change in longitude and latitude
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # This is the main formula (a lot of sine and cosine!)
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    
    distance = Earth_Radius_km * c
    return distance

print("Step 3: Calculating real distance (Haversine)...")
# I use a loop-like function (.apply) to run my distance function for every row.
my_data_table['distance_km'] = my_data_table.apply(
    lambda row: calculate_haversine_distance(
        row['pickup_lat'], row['pickup_lon'],
        row['dropoff_lat'], row['dropoff_lon']
    ),
    axis=1
)
print("New distances added:")
print(my_data_table[['trip_id', 'distance_km']].head())
print("\n")


# --- 4. Normalization Method 2: Geometric Projection (UTM) ---
# This is where I convert the degrees (on a globe) into meters (on a flat map).
# This is great because distance calculations are now simple!

print("Step 4: Converting coordinates from Globe (degrees) to Flat Grid (meters)...")

# 4a. GeoPandas needs the coordinates in a special format (Point object)
# I combine the lon and lat columns for the pickup points.
geometry_list = [Point(xy) for xy in zip(my_data_table['pickup_lon'], my_data_table['pickup_lat'])]

# 4b. Create the GeoDataFrame and tell it the current system is WGS 84 (EPSG:4326)
my_geo_data = geopandas.GeoDataFrame(my_data_table, geometry=geometry_list, crs='EPSG:4326')

# 4c. Now, transform the data to the flat map system (UTM Zone 18N for NYC: EPSG:32618)
projected_data = my_geo_data.to_crs(epsg=32618)

# 4d. Put the new X (east/west) and Y (north/south) coordinates back into my main table.
my_data_table['pickup_x_meters'] = projected_data.geometry.x
my_data_table['pickup_y_meters'] = projected_data.geometry.y

print("New Projected X and Y coordinates (in meters):")
print(my_data_table[['trip_id', 'pickup_x_meters', 'pickup_y_meters']].head())
print("\n")


# --- 5. Normalization Method 3: Spherical Sin/Cos Encoding ---
# This fixes the problem where longitude (-180 degrees and +180 degrees) wraps around.
# I turn each coordinate into two numbers (sine and cosine).

print("Step 5: Applying the Sin/Cos trick for spherical data...")

# I need the pickup coordinates in radians again.
my_data_table['lon_rad'] = np.deg2rad(my_data_table['pickup_lon'])
my_data_table['lat_rad'] = np.deg2rad(my_data_table['pickup_lat'])

# Now calculate the sine and cosine for both latitude and longitude.
my_data_table['pickup_lon_sin'] = np.sin(my_data_table['lon_rad'])
my_data_table['pickup_lon_cos'] = np.cos(my_data_table['lon_rad'])
my_data_table['pickup_lat_sin'] = np.sin(my_data_table['lat_rad'])
my_data_table['pickup_lat_cos'] = np.cos(my_data_table['lat_rad'])

print("New Sin/Cos features created:")
print(my_data_table[['trip_id', 'pickup_lon_sin', 'pickup_lon_cos', 'pickup_lat_sin', 'pickup_lat_cos']].head())
print("\n")


# --- 6. Final Normalization Step: Min-Max Scaling ---
# This squishes the large 'meter' numbers down to a small range (0 to 1).
# This is super important so that one feature doesn't unfairly dominate an algorithm.

print("Step 6: Squishing the meter values to a 0-to-1 range (Min-Max Scaling)...")

# Initialize the scaler object
my_scaler = MinMaxScaler()

# I apply the scaler ONLY to the projected X and Y meter columns.
my_data_table[['pickup_x_scaled', 'pickup_y_scaled']] = my_scaler.fit_transform(
    my_data_table[['pickup_x_meters', 'pickup_y_meters']]
)

print("Final Scaled X and Y values:")
print(my_data_table[['trip_id', 'pickup_x_scaled', 'pickup_y_scaled']].head())
print("\n")


# --- 7. Final Project Output: The Resulting Table ---
print("--- PROJECT COMPLETE: Final Processed Data Table ---")

# I'll only show the key columns to demonstrate the transformation
final_output = my_data_table[[
    'trip_id', 
    'pickup_lat', 'pickup_lon', 
    'distance_km', 
    'pickup_x_meters', 'pickup_y_meters', 
    'pickup_lon_sin', 'pickup_lon_cos',
    'pickup_x_scaled', 'pickup_y_scaled'
]]

print(final_output)

# I can see how the four original numbers are now many more,
# and they are all better formatted for a computer to learn from!
