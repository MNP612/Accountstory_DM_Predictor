import pandas as pd
import numpy as np
import datetime
import re
import os
import warnings

warnings.simplefilter(action='ignore')

my_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'model'))

def process_data(data):
    data = pd.Series(data)
    
    useless_features = ['id', 'full_name', 'middle_initial', 'middle_name', 'last_name', 'last_initial'
                        , 'birth_date', 'linkedin_url',
        'linkedin_username', 'linkedin_id', 'facebook_url', 'facebook_username',
        'facebook_id', 'twitter_url', 'twitter_username', 'github_url',
        'github_username', 'work_email', 'personal_emails', 'mobile_phone',
        'industry', 'job_company_id', 'job_company_name',
        'job_company_website', 'job_company_size', 'job_company_founded',
        'job_company_industry', 'job_company_linkedin_url',
        'job_company_linkedin_id', 'job_company_facebook_url',
        'job_company_twitter_url', 'job_company_location_name',
        'job_company_location_locality', 'job_company_location_metro',
        'job_company_location_region', 'job_company_location_geo',
        'job_company_location_street_address',
        'job_company_location_address_line_2',
        'job_company_location_postal_code', 'job_company_location_country',
        'job_company_location_continent', 'job_last_updated',
        'location_name', 'location_locality', 'location_metro',
        'location_region', 'location_country', 'location_continent',
        'location_street_address', 'location_address_line_2',
        'location_postal_code', 'location_geo', 'location_last_updated',
        'phone_numbers', 'emails', 'interests', 'location_names',
        'regions', 'countries', 'street_addresses',
        'profiles', 'version_status'
                    ]
    
    data.drop(useless_features, inplace=True)
    
    # engineer 'Name'
    us_names = pd.read_csv(my_path + '/us_names.csv')
    us_names = us_names['name'].str.lower().to_list()
    if data['first_name'] in us_names:
        data['us_name'] = 1
    else:
        data['us_name'] = 0
            
    data.drop('first_name', inplace=True) # drop 'name' column

    # gender string to integer
    if data['gender'] == 'male':
        data['gender'] = 1
    elif data['gender'] == 'female':
        data['gender'] = -1
    elif data['gender'] == 'None' or data['gender'] == None:
        data['gender'] = 0

    # engineer 'Uni'
    uni_rank = pd.read_csv(my_path + '/uni_rank.csv', sep='\t')[['World Rank', 'Institution']].to_dict()['Institution']

    def string_replace(origignal_string):
        origignal_string = origignal_string.lower()
        str_lst = ['\n', ',', '\xa0', 'university', 'of california', 'of ', '-', '–', ' ']
        
        for string in str_lst:
            origignal_string = origignal_string.replace(string, '')
        
        origignal_string = re.sub(r"\([^()]*\)", "", origignal_string) # remove parantheses
        origignal_string = re.sub(r"^\s+", "", origignal_string) # remove needing space
        origignal_string = re.sub(r"\s+$", "", origignal_string) # remove trailing space
        return origignal_string

    uni_rank = {string_replace(y):x for x,y in uni_rank.items()}
    uni_rank['mit'] = uni_rank['massachusettsinstitutetechnology']
    uni_rank['caltec'] = uni_rank['californiainstitutetechnology']

    
    n = len(uni_rank)
    person_uni_rank = 0
    for school in data['education']:
        school_name = string_replace(school['school']['name'])
        try:
            if school_name in uni_rank:
                person_uni_rank +=  n - uni_rank[school_name]        
        except:
            pass
    
    data['uni_rank'] = person_uni_rank

    # engineer 'degrees'
    degree_score = 0
    for school in data['education']:
            if school['degrees']:
                degree_score += 1
    
    data['degree_score'] = degree_score

    # one-hot 'major'
    for school in data['education']:
        for major in school['majors']:
            maj = 'edu_' + major
            if maj not in data:
                data[maj] = 0
            data[maj] = 1
    data.drop('education', inplace=True)

    # drop 'birth year', 'job_title'
    data.drop('birth_year', inplace=True)
    data.drop('job_title', inplace=True)

    # calculate month of service
    try:
        date = datetime.datetime.strptime(data['job_start_date'], '%Y-%m')
        mos = 12 * (datetime.datetime.now().year - date.year) + (datetime.datetime.now().month - date.month)
        
    except:
        mos = 40 # average mos
    
    data['month_of_service'] = mos
    data.drop('job_start_date', inplace=True)

    # job_title_levels (one-hot)
    unique_attributes = []
    for elem in data['job_title_levels']:
        string = 'level_' + elem
        if string not in unique_attributes:
            data[string] = 1

    data.drop('job_title_levels', inplace=True)

    # number of skills
    skill_cnt = len(data['skills'])
    if skill_cnt > 0:
        data['skills_count'] = skill_cnt
    else: data['skills_count'] = 40 # average skill_cnt

    data.drop('skills', inplace=True)
   

    # for experience just count the number of different companies.
    # add feature that indicates if person worked at a global 10001+ employee company
    data['company_global'] = 0
    data['company_count'] = len(data['experience'])
        
    try:
        if data['company']['size'] == '10001+':
            data['company_global'] = 1
    except:
        data['company_global'] = 0

    data.drop('experience',  inplace=True)
    
    
    

    # one-hot job_title_roles
    try:
        data['role_' + data['job_title_role']] = 1
    except:
        pass
    data.drop('job_title_role',  inplace=True)

    try:
        data['sub_role_' + data['job_title_sub_role']] = 1
    except:
        pass
    data.drop('job_title_sub_role',  inplace=True)

    # cast into important feature to make full dataset
    all_features = pd.read_csv(my_path + '/feat_importance.csv')['feature'].to_list()
    final_data = pd.DataFrame()
    for feat in all_features:
        if feat in data:
            final_data[feat] = [data[feat]]
        else:
            final_data[feat] = [0]

    # scale data
    std = np.load(model_path + '/std.npy')
    mean = np.load(model_path + '/mean.npy')
    final_data = (final_data.to_numpy() - mean) / std

    return final_data