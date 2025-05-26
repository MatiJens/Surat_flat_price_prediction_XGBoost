import re

import numpy as np

def sort_row_by_schema(row, columns_order):
    pattern_name = re.compile(r'.*Surat.*', re.IGNORECASE)
    valid_area_type = ['Built Area', 'Carpet Area', 'Plot Area', 'Super Area']
    pattern_square = re.compile(r'^\d+\s*(sqft|sqyrd|sqm)\b$', re.IGNORECASE)
    valid_transaction = ['New Property', 'Resale']
    pattern_status = re.compile(r'(Ready to Move|Poss\. by.*)', re.IGNORECASE)
    pattern_floor = re.compile(r'.*out of.*', re.IGNORECASE)
    valid_furnishing = ['Unfurnished', 'Semi-Furnished', 'Furnished', 'Yes', 'No']
    pattern_facing = re.compile(r'(North.*)|(South.*)|(West.*)|(East.*)')
    pattern_price_per = re.compile(r'₹\d(,\d+)*\s+per\s+sqft\s*',
                                   re.IGNORECASE)
    pattern_price = re.compile(r'₹\d+(\.\d+)*\s+(Lac|Cr)\s*',
                               re.IGNORECASE)

    schema = {
        'property_name' : {'type' : 'regex', 'pattern' : pattern_name},
        'areaWithType': {'type' : 'list', 'options' : valid_area_type},
        'square_feet' : {'type' : 'regex', 'pattern' : pattern_square},
        'transaction' : {'type': 'list', 'options': valid_transaction},
        'status' : {'type': 'regex', 'pattern': pattern_status},
        'floor' : {'type': 'regex', 'pattern': pattern_floor},
        'furnishing' : {'type': 'list', 'options': valid_furnishing},
        'facing' : {'type': 'regex', 'pattern': pattern_facing},
        'price_per_sqft' : {'type': 'regex', 'pattern': pattern_price_per},
        'price': {'type': 'regex', 'pattern': pattern_price}
    }

    value_pool = list(row.values)

    found_by_type = dict.fromkeys(columns_order, np.nan)

    for value in list(value_pool):
        for type_name, rules in schema.items():
            is_match = False

            if rules['type'] == 'regex' and rules['pattern'].fullmatch(str(value)):
                is_match = True
            elif rules['type'] == 'list' and value in rules['options']:
                is_match = True

            if is_match:
                found_by_type.update({type_name : value})
                value_pool.remove(value)
                break

    ordered_values = []
    for value in found_by_type.values():
        ordered_values.append(value)

    return ordered_values