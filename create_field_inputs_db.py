import pandas as pd
import numpy as np
import datetime
import glob
from tqdm import tqdm


def closest_date(date_list, date):
    return min(date_list, key=lambda x: abs(x - date))


path = "Field_Inputs/csv fields"

all_csvs = glob.iglob(path+ "/*.csv")

field_df_dict = {}
field_id_dict = {}

for csv in all_csvs:
    df = pd.read_csv(csv)
    FieldDefinedName = df["Field.Defined.Name"][0]
    try:
        field_name, field_id = FieldDefinedName.split('.')
    except ValueError:
        if FieldDefinedName.replace(' ','').isalpha():
            field_name = FieldDefinedName
            field_id = field_name
        else:
            field_name_i,field_name_ii,field_id = FieldDefinedName.split('.')
            field_name = ' '.join([field_name_i,field_name_ii])

    field_id_dict[field_name] = field_id
    field_df_dict[field_id] = df


def field_closest_date(field_id,date):
    field_df = field_df_dict[field_id]
    field_dates = pd.to_datetime(field_df["Actual.Issued.Date"]).dt.date.tolist()
    closest = closest_date(field_dates,date).strftime("%d/%m/%Y")
    return closest


def field_closest_product_2_date(field_name_or_id,date,name=True):
    if name:
        field_id = field_id_dict[field_name_or_id]
    else:
        field_id = field_name_or_id

    date = datetime.datetime.strptime(date,"%d/%m/%Y").date()
    closest_date = field_closest_date(field_id,date)
    field_df = field_df_dict[field_id]
    closest_date_row = field_df.loc[field_df["Actual.Issued.Date"] == closest_date]

    product = closest_date_row["Product.Name"].values[0]
    rate_per_ha = closest_date_row["Rate.per.Application.Area.ha"].values[0]
    units = closest_date_row["Units"].values[0]

    return closest_date, product, rate_per_ha, units


date_applied, product, rate_per_ha, units = field_closest_product_2_date("Pet Paddock","10/5/2018")

print(f"{rate_per_ha} {units} per ha of {product} was applied on {date_applied}")






























