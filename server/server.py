from flask import Flask, request, jsonify
import util

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def get_predict():
    return "Hello"


if __name__ == '__main__':
    print("Starting Python Flask server for Colombo Land Price Prediction")
    # app.run()
    api_key = "YOUR_GOOGLE_MAPS_API_KEY"
    location = "37.7749,-122.4194"  # Example location (latitude, longitude)
    radius = 2000  # Example radius in meters

    # Example for Government Schools A
    count_govt_schools_A = get_count_of_school_type(api_key, location, radius, 'government')
    nearest_location_govt_schools_A, distance_govt_schools_A = get_nearest_school_type(api_key, location, radius,
                                                                                       'government')

    # Example for Government Schools B
    count_govt_schools_B = get_count_of_school_type(api_key, location, radius, 'government B')
    nearest_location_govt_schools_B, distance_govt_schools_B = get_nearest_school_type(api_key, location, radius,
                                                                                       'government B')

    # Example for Semigovernment Schools
    count_semigovt_schools = get_count_of_school_type(api_key, location, radius, 'semigovernment')
    nearest_location_semigovt_schools, distance_semigovt_schools = get_nearest_school_type(api_key, location, radius,
                                                                                           'semigovernment')

    # Example for International Schools
    count_intl_schools = get_count_of_school_type(api_key, location, radius, 'international')
    nearest_location_intl_schools, distance_intl_schools = get_nearest_school_type(api_key, location, radius,
                                                                                   'international')

    # Print results
    print(f"Count of Government Schools A: {count_govt_schools_A}")
    print(f"Nearest Government School A Location: {nearest_location_govt_schools_A}")
    print(f"Distance to Nearest Government School A: {distance_govt_schools_A}")
    print("------")
    print(f"Count of Government Schools B: {count_govt_schools_B}")
    print(f"Nearest Government School B Location: {nearest_location_govt_schools_B}")
    print(f"Distance to Nearest Government School B: {distance_govt_schools_B}")
    print("------")
    print(f"Count of Semigovernment Schools: {count_semigovt_schools}")
    print(f"Nearest Semigovernment School Location: {nearest_location_semigovt_schools}")
    print(f"Distance to Nearest Semigovernment School: {distance_semigovt_schools}")
    print("------")
    print(f"Count of International Schools: {count_intl_schools}")
    print(f"Nearest International School Location: {nearest_location_intl_schools}")
    print(f"Distance to Nearest International School: {distance_intl_schools}")
