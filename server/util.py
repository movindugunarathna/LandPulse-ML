from datetime import date
import math
import json
import pickle
import threading

import pandas as pd
import requests

__schools = None
__city_dic = None
__data_columns = {}
__model = None
API_KEY = "AIzaSyBYMGxceM10RqSBpWvVRwmL9u_lyjRYb88"


def get_input(location, radius):
    column_names = [
        "govtscl_a_count", "govtscl_a_mindist", "govtscl_b_count", "govtscl_b_mindist",
        "semigovtscl_count", "semigovtscl_mindist", "intlscl_count", "intlscl_mindist",
        "uni_count", "uni_mindist", "express_mindist", "railway_mindist", "bank_mindist",
        "banks_2km_count", "financeco_mindist", "financecos_2km_count", "govt_hospital_mindist",
        "govt_hospitals_count", "pvt_hospital_mindist", "pvt_hospitals_count",
        "pvt_medcenter_mindist", "pvt_medcenters_count", "supermarket_mindist",
        "supermarkets_2km_count", "fuelstation_mindist", "fuelstations_2km_count",
        "latitude", "longitude", "curr_month", "curr_year"
    ]

    latitude, longitude = map(float, location.split(','))

    x1_df = pd.DataFrame(generate_data_object(API_KEY, location, radius), index=[0])

    # Create x2_df with additional features
    x2_df = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude],
        "curr_month": [date.today().month],
        "curr_year": [date.today().year]
    })

    # Concatenate x1_df and x2_df along columns
    x = pd.concat([x1_df, x2_df], axis=1)

    input_values = []

    for name in column_names:
        input_values.append(x[name].values[0])

    return input_values, column_names


def get_estimated_price(location, radius):
    load_saved_artifacts()

    input_values, column_names = get_input(location, radius)

    # Assuming __model is defined somewhere in your code
    output_array = __model.predict([input_values])[0]
    return {
        "Price": (output_array[0]),
        "min_next_month": (output_array[1]),
        "max_next_month": (output_array[2])
    }


def load_saved_artifacts():
    print("Loading saved artifacts...")
    global __schools
    global __data_columns
    global __city_dic
    global __model

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __schools = __data_columns[2:9]

    with open("./artifacts/city_dict.json", 'r') as f:
        __city_dic = json.load(f)['city_dict']

    with open("./artifacts/Model.pickle", 'rb') as f:
        __model = pickle.load(f)
        print("Loading saved artifacts...done!")


def get_schools():
    return __schools


def get_cities():
    return __city_dic


def generate_data_object(api_key, location_details, area_radius):
    __gen_data = {}

    with open("./artifacts/types.json", 'r') as f:
        __types = json.load(f)

    # List to store threads
    threads = []

    for category, subcategories in __types.items():
        # Use threading for concurrent execution of get_info
        thread = threading.Thread(target=process_category,
                                  args=(__gen_data, __types, api_key, location_details, area_radius, category))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    return __gen_data


def process_category(__gen_data, __types, api_key, location_details, area_radius, category):
    count_types, min_distance = get_info(api_key, location_details, area_radius, category)

    for subcategory, value in __types[category].items():
        if "_count" in subcategory:
            __types[category][subcategory] = count_types
            __gen_data[subcategory] = float(count_types)
        if "_mindist" in subcategory:
            __types[category][subcategory] = min_distance
            __gen_data[subcategory] = float(min_distance.replace("km", "").replace(" ", "")) * 1000


def get_info(api_key, location_d, radius_m, place_type):
    # Google Places API endpoint for text search
    places_endpoint = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    places_params = {
        'location': location_d,
        'radius': radius_m,
        'type': place_type,
        'key': api_key,
    }

    places_response = requests.get(places_endpoint, params=places_params)
    places_data = places_response.json()

    # Process the response
    if places_data['status'] == 'OK':
        count_types = len(places_data['results'])

        if count_types > 0:
            # Initialize variables for nearest location and minimum distance
            nearest_location = None
            min_distance = float('inf')

            for result in places_data['results']:
                location_details = result['geometry']['location']

                # Calculate distance between the given location and the current place
                current_distance = haversine_distance(location_d, location_details)

                # Update nearest location if the current distance is smaller
                if current_distance < min_distance:
                    min_distance = current_distance
                    nearest_location = location_details

            if nearest_location is not None:
                # Google Distance Matrix API endpoint
                distance_endpoint = "https://maps.googleapis.com/maps/api/distancematrix/json"

                # Parameters for the API request to calculate distance
                distance_params = {
                    'origins': location_d,
                    'destinations': f'{nearest_location["lat"]},{nearest_location["lng"]}',
                    'key': api_key,
                }

                # Make the API request to calculate distance
                distance_response = requests.get(distance_endpoint, params=distance_params)
                distance_data = distance_response.json()

                # Process the distance response
                if distance_data['status'] == 'OK':
                    if 'distance' in distance_data['rows'][0]['elements'][0]:
                        distance = distance_data['rows'][0]['elements'][0]['distance']['text']
                        return count_types, distance
                    else:
                        print(f"Distance information not available for {place_type}")
                        return count_types, "0"
                else:
                    print(f"Error calculating distance for {place_type}: {distance_data['status']}")
                    return count_types, "0"
            else:
                print("Error finding nearest location.")
                return count_types, "0"
        else:
            print(f"No {place_type} found.")
            return 0, "0"
    else:
        print(f"Error finding {place_type}: {places_data['status']}")
        return 0, "0"


def haversine_distance(origin, destination):
    # Calculate the Haversine distance between two points on the Earth
    lat1, lon1 = map(float, origin.split(","))
    lat2, lon2 = destination["lat"], destination["lng"]

    R = 6371  # Earth radius in kilometers

    d_lat = deg2rad(lat2 - lat1)
    d_lon = deg2rad(lon2 - lon1)

    a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) +
         math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) *
         math.sin(d_lon / 2) * math.sin(d_lon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c  # Distance in kilometers
    return distance


def deg2rad(deg):
    return deg * (math.pi / 180)


if __name__ == '__main__':
    location_input = "6.897928711019126,79.91887213019518"
    radius_input = 5000
    x = get_estimated_price(location_input, radius_input)
