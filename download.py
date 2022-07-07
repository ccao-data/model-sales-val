import pandas as pd
from sodapy import Socrata
import csv


class DataPortalCollector: 

    def __init__(self):
        self.client = Socrata("datacatalog.cookcountyil.gov", 'BVq8TXDxCs4MoWS2nu27YxWtB', timeout=90)
        self.COLS = 'distinct on (pin) pin, township_code, year, class, census_tract_geoid, lat, lon'
        self.csv_cols = ['pin', 'township_code', 'year', 'class', 'census_tract_geoid',
                         'lat', 'lon']
        self.query = """SELECT pin, township, sale_date, census_block_geoid, census_puma_geoid, lat, lon, year
                        WHERE year IN (2022, 2021, 2020, 2019)"""

    def find_records(self): 
        with open('parcels_geoinfo.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_cols)
            writer.writeheader()
            data_dict = self.client.get_all("tx2p-k2g9", select=self.COLS, where='year in (2022, 2021, 2020, 2019)')
            writer.writerows(data_dict)

if __name__ == "__main__":

    downloader = DataPortalCollector()
    downloader.find_records()
