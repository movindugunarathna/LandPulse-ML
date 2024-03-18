from datetime import date, timedelta
from typing import Type, Union

import googlemaps
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import math
import json
import pickle
import threading
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

__data_columns = ()
__land_types = {}
__model: Union[None, Type[MultiOutputRegressor]] = None
API_KEY = os.getenv("API_KEY")
gmaps = googlemaps.Client(key=API_KEY)
__lock = threading.Lock()
min_date = date.today()
# Categories for one-hot encoding
categories = [
    'Agricultural',
    'Commercial',
    'Residential',
    'Other'
]


def load_saved_artifacts():
    try:
        print("Loading saved artifacts...")
        global min_date
        global __data_columns
        global __model
        global __land_types
        global gmaps

        min_date = datetime.strptime("09/10/2015", "%m/%d/%Y")

        with open("./artifacts/columns.json", 'r') as f:
            __data_columns = tuple(json.load(f)['data_columns'])

        with open("./artifacts/landType.json", 'r') as f:
            __land_types = json.load(f)

        with open("./artifacts/Model.pickle", 'rb') as f:
            __model = pickle.load(f)
            if not isinstance(__model, MultiOutputRegressor):
                raise ValueError("Loaded model is not a MultiOutputRegressor.")
            base_estimator = __model.estimator
            if not isinstance(base_estimator, RandomForestRegressor):
                raise ValueError("Base estimator of MultiOutputRegressor is not a RandomForestRegressor.")
        print("Loading saved artifacts...done!")
    except Exception as e:
        print("load_saved_artifacts err" + str(e))


def get_estimated_price(location, land_type, radius):
    try:
        results = get_input(location, land_type, radius)
        input_values = results['input_values']
        __obj_dic = results['__obj_dic']

        while True:
            if len(input_values) == len(__data_columns):
                try:
                    if __model is None:
                        raise ValueError("Model is not loaded. Cannot make predictions.")

                    return_obj = {
                        "Obj": __obj_dic
                    }

                    for i in range(-4, 2):
                        current_date = date.today() + timedelta(days=i * 365)
                        date_from = add_date_count(current_date, min_date)

                        # add month randomly
                        input_values[len(input_values) - 1] = date_from

                        output_array = __model.predict(X=[input_values])[0]

                        return_obj[current_date.year] = {
                            "min_next": float(output_array[1]),
                            "max_next": float(output_array[2]),
                            "price": float(output_array[0]),
                            "year": current_date.year
                        }
                        if i == 0:
                            return_obj["price"] = float(output_array[0])

                    return return_obj

                except Exception as e:
                    print(e)
                    return {"message": f"Error: {str(e)}"}
            if len(input_values) != len(__data_columns):
                print("Columns do not match")
                print("Columns:", str(__data_columns))
                print("input_values:", str(input_values))
                break

        return {"message": "Nothing came out!"}
    except Exception as e:
        print("get_estimated_price err" + str(e))


def get_air_quality(latitude, longitude, api_key):
    try:
        url = 'https://airquality.googleapis.com/v1/currentConditions:lookup?key=' + api_key

        # Define the data to be sent in the request body (in JSON format)
        data = {
            "universalAqi": True,
            "location": {
                "latitude": latitude,
                "longitude": longitude,
            },
            "extraComputations": [
                "HEALTH_RECOMMENDATIONS",
                "DOMINANT_POLLUTANT_CONCENTRATION",
                "POLLUTANT_CONCENTRATION",
                "LOCAL_AQI",
                "POLLUTANT_ADDITIONAL_INFO"
            ],
            "languageCode": "en"
        }

        response = requests.post(url, json=data)
        data = response.json()

        value = 0
        if 'indexes' in data:
            value = sum(index['aqi'] for index in data['indexes'])

        return value
    except Exception as e:
        print("get_air_quality err" + str(e))


def get_input(location, land_type, radius):
    try:
        latitude, longitude = map(float, location.split(','))
        land_type_encoded = land_type_generation(land_type)
        date_from = add_date_count(date.today(), min_date)

        generated_obj1 = generate_data_object(location, radius)
        generated_obj2 = {
            "agricultural": land_type_encoded[0],
            "commercial": land_type_encoded[1],
            "residential": land_type_encoded[2],
            "other": land_type_encoded[3],
            "lat": latitude,
            "long": longitude,
            "air": get_air_quality(latitude, longitude, API_KEY),
            "date_from": date_from,
        }

        # Create DataFrames from the generated objects
        df1 = pd.DataFrame([generated_obj1])
        df2 = pd.DataFrame([generated_obj2])

        # Concatenate the DataFrames along the columns axis
        connected_df = pd.concat([df1, df2], axis=1).reindex(columns=__data_columns)

        # Update generated_obj2 with generated_obj1
        generated_obj2.update(generated_obj1)
        __obj_dic = generated_obj2.copy()

        return {
            'input_values': connected_df.values.tolist()[0],  # Convert DataFrame to list of values
            '__obj_dic': __obj_dic
        }
    except Exception as e:
        print("get_input err" + str(e))


def generate_data_object(location_details, area_radius):
    try:
        __gen_data = {}
        with open("./artifacts/types.json", 'r') as f:
            __types = json.load(f)

        threads = []

        for place in __types:
            thread = threading.Thread(target=process_category,
                                      args=(__gen_data, __types, location_details, area_radius, place))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        return __gen_data
    except Exception as e:
        print("generate_data_object err" + str(e))


def process_category(__gen_data, __types, location_details, area_radius, category):
    try:
        count_types, min_distance = get_info(location_details, area_radius, category)
        print("Initialized " + category + " : " + str(count_types) + " , " + str(min_distance))

        with __lock:
            __gen_data[category + '_count'] = float(count_types)
            __gen_data[category + '_mdist'] = float(min_distance)
    except Exception as e:
        print("process_category err" + str(e))


def get_info(location_det, radius, type_de):
    try:
        lat, lng = location_det.split(',')
        places = gmaps.places_nearby(location=(lat, lng), radius=radius, type=type_de)
        count = len(places['results'])

        if count > 0:
            distances = [haversine_distance(location_det, place['geometry']['location']) for place in places['results']]
            if len(distances) > 0:  # Check if distances list is not empty
                closest_place_index = distances.index(min(distances))
                closest_place = places['results'][closest_place_index]
                min_distance = gmaps.distance_matrix(origins=(lat, lng), destinations=(
                    closest_place['geometry']['location']['lat'], closest_place['geometry']['location']['lng']))[
                    'rows'][0]['elements'][0][
                    'distance']['value']
            else:
                min_distance = 0
        else:
            min_distance = 0
        return count, min_distance

    except googlemaps.exceptions.ApiError as e:
        print(f"An API error occurred while processing {type_de}: {e}")
        return 0, 0
    except ValueError as ve:
        print(f"Error in distance calculation: {ve}")
        return 0, 0
    except Exception as ex:
        print(f"An unexpected error occurred while processing {type_de}: {ex}")
        return 0, 0


def haversine_distance(origin, destination):
    try:
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
    except Exception as e:
        print("haversine_distance err" + str(e))


def land_type_generation(land_type):
    try:
        one_hot_encoding = [0] * len(categories)

        # Split the row string into categories
        row_categories = land_type.replace(" ", "").split(',')

        # Update one-hot encoding based on presence of categories
        for category in row_categories:
            if category in categories:
                index = categories.index(category)
                one_hot_encoding[index] = 1

        return one_hot_encoding
    except Exception as e:
        print("land_type_generation err" + str(e))


def deg2rad(deg):
    try:
        return deg * (math.pi / 180)
    except Exception as e:
        print("deg2rad err" + str(e))


def date_to_numeric(date_str):
    try:
        # Convert the date string to a numerical representation
        date_transferred = datetime.strptime(date_str, "%m/%d/%Y")
        return date_transferred.timestamp()
    except Exception as e:
        print("date_to_numeric err" + str(e))


def numeric_to_date(numeric_value):
    try:
        # Convert the numerical representation back to the original date string
        date_transferred = datetime.fromtimestamp(numeric_value)
        return date_transferred.strftime("%m/%d/%Y")

    except Exception as e:
        print("numeric_to_date err" + str(e))


def calculate_month(current_month, increment):
    try:
        # Adjust the month and year accordingly
        year_offset = (current_month + increment - 1) // 12
        new_month = (current_month + increment - 1) % 12 + 1
        new_year = date.today().year + year_offset

        return new_year, new_month
    except Exception as e:
        print("calculate_month err" + str(e))


def add_date_count(given_date, min_date_val):
    try:
        # Convert min_date to datetime.date object
        min_date_val = min_date_val.date()

        # Calculate the difference
        difference = given_date - min_date_val

        # Extract the number of days from the difference
        return difference.days
    except Exception as e:
        print("Add date count err" + str(e))


def convert_int_to_str(d):
    if isinstance(d, dict):
        for key, value in list(d.items()):
            if isinstance(value, dict):
                convert_int_to_str(value)
            elif isinstance(value, int) or isinstance(value, float):
                d[key] = str(value)
            if isinstance(key, int):
                d[str(key)] = d.pop(key)
    return d
