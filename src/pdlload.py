import requests
import os

my_path = os.path.abspath(os.path.dirname(__file__))

def call_enrich_api(company_and_name):
    API_KEY = open(my_path + '/PDL_api_key.txt', 'r').readlines(0)

    pdl_url = "https://api.peopledatalabs.com/v5/person/enrich"

    params = {
        "api_key": API_KEY,
        "company": [company_and_name.split()[2]],   
        "first_name": [company_and_name.split()[0]],
        "last_name": [company_and_name.split()[1]]
    }

    json_response = requests.get(pdl_url,  params=params).json()

    if json_response["status"] == 200:
        return json_response['data']

    else:
        print("Enrichment unsuccessful. See error and try again.")
        print("error:", json_response)

