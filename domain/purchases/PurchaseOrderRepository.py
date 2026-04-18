import os
import pandas as pd

pd.set_option('display.width', 600)
pd.set_option('display.max_columns', 20)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PurchaseOrderRepository:
    """Class for data persistence"""

    @staticmethod
    def load_purchase_order_csv():
        """Load data file in CSV format"""
        try:
            return pd.read_csv(
                os.path.join(BASE_DIR, "data", "purchase-order-data-2012-2015-.csv"),
                delimiter=",",
                encoding="utf-8"
            )
        except Exception as e:
            print("Error loading data file: " + e.__str__())
            raise

    @staticmethod
    def load_purchase_order_parquet():
        """Load data file in Parquet format"""
        try:
            return pd.read_parquet(
                os.path.join(BASE_DIR, "data", "purchase-order-data-2012-2015-.parquet")
            )
        except Exception as e:
            print("Error loading parquet file: " + e.__str__())
            raise

# print(PurchaseOrderRepository.load_purchase_order_csv())