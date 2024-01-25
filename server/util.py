from datetime import date
from typing import Type, Union
from sklearn.tree import DecisionTreeRegressor
import math
import json
import pickle
import threading
import pandas as pd
import requests

__data_columns = []
__model: Union[None, Type[DecisionTreeRegressor]] = None
API_KEY = "AIzaSyBYMGxceM10RqSBpWvVRwmL9u_lyjRYb88"
__lock = threading.Lock()


def load_saved_artifacts():
    print("Loading saved artifacts...")
    global __data_columns
    global __model

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']

    with open("./artifacts/Model.pickle", 'rb') as f:
        __model = pickle.load(f)
        if not isinstance(__model, DecisionTreeRegressor):
            raise ValueError("Loaded model is not a DecisionTreeRegressor.")
    print("Loading saved artifacts...done!")


def get_estimated_price(location, radius):
    input_values = get_input(location, radius)

    while True:

        if len(input_values) == len(__data_columns):
            try:
                if __model is None:
                    raise ValueError("Model is not loaded. Cannot make predictions.")

                dictionary = {}
                for i, name in enumerate(__data_columns):
                    dictionary[name] = input_values[i]

                print("Dictionary: "+str(dictionary))

                output_array = __model.predict(X=[input_values])[0]
                return {
                    "price": float(output_array[0]),
                    "min_next": float(output_array[1]),
                    "max_next": float(output_array[2]),
                    "Obj": {key: str(value) for key, value in dictionary.items()}
                }

            except Exception as e:
                return {"message": f"Error: {str(e)}"}

        if len(input_values) != len(__data_columns): break

    return {"message": "Nothing came out!"}


def get_input(location, radius):
    latitude, longitude = map(float, location.split(','))

    x1_df = pd.DataFrame(generate_data_object(API_KEY, location, radius), index=[0])

    x2_df = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude],
        "curr_month": [date.today().month],
        "curr_year": [date.today().year]
    })

    connected_df = pd.concat([x1_df, x2_df], axis=1)

    return [connected_df[name].values[0] for name in __data_columns]


def generate_data_object(api_key, location_details, area_radius):
    __gen_data = {}
    with open("./artifacts/types.json", 'r') as f:
        __types = json.load(f)

    threads = []

    for category, subcategories in __types.items():
        thread = threading.Thread(target=process_category,
                                  args=(__gen_data, __types, api_key, location_details, area_radius, category))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return __gen_data


def process_category(__gen_data, __types, api_key, location_details, area_radius, category):
    count_types, min_distance = get_info(api_key, location_details, area_radius, category)

    with __lock:

        for subcategory, value in __types[category].items():
            if "_count" in subcategory:
                __types[category][subcategory] = count_types
                __gen_data[subcategory] = float(count_types)
            if "_mindist" in subcategory:
                if "km" in min_distance:
                    __types[category][subcategory] = min_distance
                    __gen_data[subcategory] = float(min_distance.replace("km", "").replace(" ", "")) * 1000
                elif "m" in min_distance:
                    __types[category][subcategory] = min_distance
                    __gen_data[subcategory] = float(min_distance.replace("m", "").replace(" ", ""))
                else:
                    __types[category][subcategory] = min_distance
                    __gen_data[subcategory] = float(min_distance.replace(" ", ""))


def get_info(api_key, location_d, radius_m, place_type):
    places_endpoint = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    places_params = {
        'location': location_d,
        'radius': radius_m,
        'type': place_type,
        'key': api_key,
    }

    places_response = requests.get(places_endpoint, params=places_params)
    places_data = places_response.json()

    if places_data['status'] == 'OK':
        count_types = len(places_data['results'])

        if count_types > 0:
            nearest_location = None
            min_distance = float('inf')

            for result in places_data['results']:
                location_details = result['geometry']['location']
                current_distance = haversine_distance(location_d, location_details)

                if current_distance < min_distance:
                    min_distance = current_distance
                    nearest_location = location_details

            if nearest_location is not None:
                distance_endpoint = "https://maps.googleapis.com/maps/api/distancematrix/json"
                distance_params = {
                    'origins': location_d,
                    'destinations': f'{nearest_location["lat"]},{nearest_location["lng"]}',
                    'key': api_key,
                }

                distance_response = requests.get(distance_endpoint, params=distance_params)
                distance_data = distance_response.json()

                if distance_data['status'] == 'OK':
                    if 'distance' in distance_data['rows'][0]['elements'][0]:
                        distance = distance_data['rows'][0]['elements'][0]['distance']['text']
                        return count_types, distance
                    else:
                        print(f"Distance information not available for {place_type}")
                        return count_types, "0 km"
                else:
                    print(f"Error calculating distance for {place_type}: {distance_data['status']}")
                    return count_types, "0 km"
            else:
                print("Error finding nearest location.")
                return count_types, "0 km"
        else:
            print(f"No {place_type} found.")
            return 0, "0 km"
    else:
        print(f"Error finding {place_type}: {places_data['status']}")
        return 0, "0 km"


def haversine_distance(origin, destination):
    lat1, lon1 = map(float, origin.split(","))
    lat2, lon2 = destination["lat"], destination["lng"]

    const_r = 6371
    d_lat = deg2rad(lat2 - lat1)
    d_lon = deg2rad(lon2 - lon1)

    a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) +
         math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) *
         math.sin(d_lon / 2) * math.sin(d_lon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = const_r * c
    return distance


def deg2rad(deg):
    return deg * (math.pi / 180)


if __name__ == '__main__':
    location_input = "6.897928711019126,79.91887213019518"
    radius_input = 5000
    x = get_estimated_price(location_input, radius_input)
    print(x)
