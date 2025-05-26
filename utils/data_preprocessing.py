import re
from email.policy import default

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from utils.sort_row_by_schema import sort_row_by_schema


def data_preprocessing(csv_path):
    # Load data
    data = pd.read_csv(csv_path)

    # Delete unnecessary columns
    data = data.drop(columns=['description'])

    # Get columns names and order
    columns_order = data.columns
    columns_order_list = columns_order.to_list()

    # Sort columns by correct values
    data = data.apply(sort_row_by_schema, axis=1, result_type='expand', columns_order=columns_order_list)

    # Assign correct columns names and index to sorted df
    data.columns = columns_order

    # Fill na in property_name and set every letter to lower
    data['property_name'] = data['property_name'].fillna('').astype(str)
    data['property_name'] = data['property_name'].str.lower()

    # Add rooms number
    BHK = data['property_name'].str.extract((r'(\d+)\s*BHK'), flags=re.IGNORECASE)
    #data['status'].str.extract(r'Poss\. by (.*)', flags=re.IGNORECASE)
    data['BHK'] = pd.to_numeric(BHK[0])

    # Add flat type as one hot encodding
    flat_type = data['property_name'].str.extract(r'.*(Apartment|Villa|Office Space|Shop|House|Builder Floor|Penthouse|Industrial Land).*', flags=re.IGNORECASE)
    flat_type = pd.get_dummies(flat_type, columns=[0], prefix='', prefix_sep='', drop_first=True, dtype=int)
    data = pd.concat([data, flat_type], axis=1)

    # Delete row with no price
    no_price_mask = (data['price_per_sqft'].notna()) | (data['price'].notna())
    data = data[no_price_mask]

    # One hot encoding to area type
    data = pd.get_dummies(data, columns=['areaWithType'], prefix='', prefix_sep='', drop_first=True, dtype=int)

    # Transforming every square to square feet (some of them are sqm or sqyrd)
    square_feet_conditions = [
        data['square_feet'].str.split().str[1] == 'sqm',
        data['square_feet'].str.split().str[1] == 'sqyrd'
    ]
    square_feet_choices = [10.76, 9]

    data['square_numeric'] =  pd.to_numeric(data['square_feet'].str.split().str[0])

    data['square_multipliers'] = np.select(square_feet_conditions, square_feet_choices, default=1.0)

    data['square_feet'] = data['square_numeric'] * data['square_multipliers']

    data = data.drop(columns=['square_numeric', 'square_multipliers'])

    # Transforming transaction into numerical value (Resale = 0, New = 1)
    data['transaction'] = np.where(data['transaction'] == 'Resale',
                                   1,
                                   0)

    # Convert status to date
    data.loc[data['status'] == 'Ready to Move', 'status_converted'] = "May '24"
    extracted_dates = data['status'].str.extract(r'Poss\. by (.*)', flags=re.IGNORECASE)
    data['status_converted'] = data['status_converted'].fillna(extracted_dates[0])

    data['status_converted'] = data['status_converted'].fillna(data['status'])

    data['status'] = pd.to_datetime(data['status_converted'], format="%b '%y", errors='coerce')

    data['year'] = data['status'].dt.year
    data['month'] = data['status'].dt.month

    data = data.drop(columns=['status', 'status_converted'])

    # Extracting flat floor and building height
    extracted_floor = data['floor'].str.extract(r'(.*) out of (.*)', flags=re.IGNORECASE)

    char_pattern = re.compile(r'[a-zA-Z]+')
    chars_to_zero_mask = extracted_floor[0].str.match(char_pattern, na=False)

    extracted_floor.loc[chars_to_zero_mask, 0] = 0

    extracted_floor[0] = pd.to_numeric(extracted_floor[0], errors='coerce')
    extracted_floor[1] = pd.to_numeric(extracted_floor[1], errors='coerce')

    data['flat_floor'] = extracted_floor[0]
    data['building_floor'] = extracted_floor[1]

    data['floor_ratio'] = data['flat_floor'] / data['building_floor']

    data = data.drop(columns=['floor'])

    # Transforming furnishing column into numeric value
    furnishing_conditions = [
        (data['furnishing'] == 'Unfurnished') | (data['furnishing'] == 'No'),
        data['furnishing'] == 'Semi-Furnished',
        (data['furnishing'] == 'Furnished') | (data['furnishing'] == 'Yes')
    ]

    furnishing_choices = [
        0,
        1,
        2
    ]

    data['furnishing'] = np.select(furnishing_conditions, furnishing_choices, default=0)

    # TODO dla kierunków kodowanie cykliczne
    data['facing'] = data['facing'].str.replace(' ', '').str.replace('-', '')

    facing_map = {
        'North' : 0.0,
        'NorthWest' : 0.785,
        'West' : 1.57,
        'SouthWest' : 2.355,
        'South' : 3.14,
        'SouthEast' : 3.925,
        'East' : 4.17,
        'NorthEast' : 5.495
    }

    data['facing'] = pd.to_numeric(data['facing'].replace(facing_map), errors='coerce')

    data['facing_sin'] = np.sin(data['facing'])
    data['facing_cos'] = np.cos(data['facing'])

    data = data.drop(columns='facing')
    """
    # Transforming price per sqft into numeric values
    extracted_price_per_sqft = data['price_per_sqft'].str.extract(r'₹(\d+,\d+).*', flags=re.IGNORECASE)

    extracted_price_per_sqft[0] = extracted_price_per_sqft[0].str.replace(',', '.')
    extracted_price_per_sqft[0] = pd.to_numeric(extracted_price_per_sqft[0], errors='coerce')

    data['price_per_sqft'] = extracted_price_per_sqft[0]
    """

    # Deleting price_per_sqft, because it causes data leaking
    data = data.drop(columns=['price_per_sqft'])

    # Transforming price into value and multiplying it by Lac or Cr
    extracted_price = data['price'].str.extract(r'₹(\d+.\d*)\s*(Lac|Cr)', flags=re.IGNORECASE)

    extracted_price[0] = pd.to_numeric(extracted_price[0], errors='coerce')

    extracted_price[0] = np.where(extracted_price[1] == 'Lac',
                                  extracted_price[0] * 100000,
                                  extracted_price[0] * 10000000)

    data['price'] = extracted_price[0]

    data = data.dropna(subset=['price'])

    mask_price_percentile_95 = data['price'] <= data['price'].quantile(0.95)
    data = data[mask_price_percentile_95]

    mask_price_percentile_01 = data['price'] > data['price'].quantile(0.05)
    data = data[mask_price_percentile_01]

    data['price'] = data['price'].round(0)

    data['price'] = np.log1p(data['price'])

    # Filtering 0 height flats
    mask_no_height = data['building_floor'] > 0
    data = data[mask_no_height]

    # Split data into features (x) and target (y)
    features_columns = ['Carpet Area', 'Plot Area', 'Super Area', 'square_feet', 'year',
                          'month', 'transaction', 'furnishing', 'flat_floor', 'building_floor',
                           'floor_ratio', 'facing_sin', 'facing_cos', 'BHK', 'builder floor',
                        'house', 'industrial land', 'office space', 'shop', 'villa']
    target_columns = ['price']

    features = data[features_columns]
    target = data[target_columns]

    return features, target