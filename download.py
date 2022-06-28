import pandas as pd
from sodapy import Socrata
import csv



class DataPortalCollector: 

    def __init__(self):
        self.client = Socrata("datacatalog.cookcountyil.gov", None)
        self.COLS = 'pin, census_tract_geoid, census_block_geoid, census_puma_geoid, lat, lon'
        self.csv_cols = ['pin', 'census_tract_geoid', 'census_block_geoid',
                         'census_puma_geoid', 'lat', 'lon']

    def find_records(self): 
        with open('parcels_geoinfo.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_cols)
            writer.writeheader()
            data_dict = self.client.get("tx2p-k2g9", limit=10000000, select=self.COLS)
            writer.writerows(data_dict)

if __name__ == "__main__":

    downloader = DataPortalCollector()
    downloader.find_records()
