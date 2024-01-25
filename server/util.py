from datetime import date
from typing import Type, Union

import googlemaps
from sklearn.tree import DecisionTreeRegressor
import math
import json
import pickle
import threading
import pandas as pd

__data_columns = []
__land_types = {}
__model: Union[None, Type[DecisionTreeRegressor]] = None
API_KEY = "AIzaSyBYMGxceM10RqSBpWvVRwmL9u_lyjRYb88"
gmaps = googlemaps.Client(key="AIzaSyBYMGxceM10RqSBpWvVRwmL9u_lyjRYb88")
__lock = threading.Lock()


def load_saved_artifacts():
    print("Loading saved artifacts...")
    global __data_columns
    global __model
    global __land_types
    global gmaps

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']

    with open("./artifacts/landType.json", 'r') as f:
        __land_types = json.load(f)

    with open("./artifacts/Model.pickle", 'rb') as f:
        __model = pickle.load(f)
        if not isinstance(__model, DecisionTreeRegressor):
            raise ValueError("Loaded model is not a DecisionTreeRegressor.")
    print("Loading saved artifacts...done!")


def get_estimated_price(location, land_type, radius):
    input_values = get_input(location, land_type, radius)

    while True:

        if len(input_values) == len(__data_columns):
            try:
                if __model is None:
                    raise ValueError("Model is not loaded. Cannot make predictions.")

                dictionary = {}
                for i, name in enumerate(__data_columns):
                    dictionary[name] = input_values[i]

                print("Dictionary: " + str(dictionary))

                output_array = __model.predict(X=[input_values])[0]
                return {
                    "price": float(output_array[0]),
                    "min_next": float(output_array[1]),
                    "max_next": float(output_array[2]),
                    "Obj": {key: str(value) for key, value in dictionary.items()}
                }

            except Exception as e:
                return {"message": f"Error: {str(e)}"}

        if len(input_values) != len(__data_columns):
            break

    return {"message": "Nothing came out!"}


def get_input(location, land_type, radius):
    latitude, longitude = map(float, location.split(','))

    x1_df = pd.DataFrame(generate_data_object(location, radius), index=[0])

    x2_df = pd.DataFrame({
        "land_type": land_type_generation(land_type),
        "latitude": [latitude],
        "longitude": [longitude],
        "curr_month": [date.today().month],
        "curr_year": [date.today().year]
    })

    connected_df = pd.concat([x1_df, x2_df], axis=1)

    return [connected_df[name].values[0] for name in __data_columns]


def generate_data_object(location_details, area_radius):
    __gen_data = {}
    with open("./artifacts/types.json", 'r') as f:
        __types = json.load(f)

    threads = []

    for category, subcategories in __types.items():
        thread = threading.Thread(target=process_category,
                                  args=(__gen_data, __types, location_details, area_radius, category))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return __gen_data


def process_category(__gen_data, __types, location_details, area_radius, category):
    count_types, min_distance = get_info(location_details, area_radius, category)
    print(category + str(count_types) + "," + str(min_distance))

    with __lock:

        for subcategory, value in __types[category].items():
            if "_count" in subcategory:
                __types[category][subcategory] = count_types
                __gen_data[subcategory] = float(count_types)
            if "_mindist" in subcategory:
                __gen_data[subcategory] = float(min_distance)


def get_info(location_det, radius, type_de):
    lat, lng = location_det.split(',')
    try:
        places = gmaps.places_nearby(location=(lat, lng), radius=radius, type=type_de)
        count = len(places['results'])

        if count > 0:
            distances = [haversine_distance(location_det, place['geometry']['location']) for place in places['results']]
            if distances:  # Check if distances list is not empty
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


def land_type_generation(land_type):
    land_type_value = 0

    for key, value in __land_types.items():
        if key in land_type:
            land_type_value += value

    return land_type_value


def deg2rad(deg):
    return deg * (math.pi / 180)
