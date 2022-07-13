import pandas as pd
from sodapy import Socrata
import csv


class DataPortalCollector: 

    def __init__(self):
        self.client = Socrata("datacatalog.cookcountyil.gov", 'BVq8TXDxCs4MoWS2nu27YxWtB', timeout=90)
        self.COLS = 'pin, card'
        self.csv_cols = ['pin', 'card']

    def find_records(self): 
        with open('cards.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_cols)
            writer.writeheader()
            data_dict = self.client.get_all("x54s-btds", select=self.COLS, where='year in (2022, 2021, 2020, 2019)')
            writer.writerows(data_dict)

if __name__ == "__main__":

    downloader = DataPortalCollector()
    downloader.find_records()
