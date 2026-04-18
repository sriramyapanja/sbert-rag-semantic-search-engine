import pandas as pd
import spacy
from nltk.corpus import stopwords
from domain.purchases.PurchaseOrderRepository import PurchaseOrderRepository

pd.set_option('display.width', 600)
pd.set_option('display.max_columns', 20)


class PurchaseOrderPreprocessingDomain:
    """Class for preprocessing and clean the dataset"""

    def __init__(self):
        self.purchaseOrderRepository = PurchaseOrderRepository()
        self.df_purchase_order = self.purchaseOrderRepository.load_purchase_order_csv()

    def set_dataframe_index(self):
        """Defines the index of the dataframe"""
        try:
            self.df_purchase_order.index += 1
        except Exception as e:
            print("Error setting index: " + e.__str__())
            raise

    @staticmethod
    def delete_columns(df):
        """Delete unused columns"""
        try:
            list_remove_columns = [
                'Purchase Date', 'Fiscal Year', 'LPA Number',
                'Requisition Number', 'Acquisition Method',
                'Sub-Acquisition Method', 'Supplier Code',
                'Supplier Qualifications', 'Supplier Zip Code',
                'Acquisition Type', 'Sub-Acquisition Type',
                'CalCard', 'Classification Codes', 'Normalized UNSPSC',
                'Commodity Title', 'Segment', 'Segment Title',
                'Location', 'Family', 'Family Title', 'REMOVE AMERISOURCE'
            ]
            df.drop(list_remove_columns, axis=1, inplace=True)
            return df
        except Exception as e:
            print("Error deleting dataframe columns: " + e.__str__())
            raise

    @staticmethod
    def rename_columns_name(df):
        """Rename columns name"""
        try:
            dict_new_columns_name = {
                'Creation Date': 'creation_date',
                'Purchase Order Number': 'purchase_order_number',
                'Department Name': 'department_name',
                'Supplier Name': 'supplier_name',
                'Item Name': 'item_name',
                'Item Description': 'item_description',
                'Quantity': 'quantity',
                'Unit Price': 'unit_price',
                'Total Price': 'total_price',
                'Class': 'class',
                'Class Title': 'class_title'
            }
            df.rename(columns=dict_new_columns_name, inplace=True)
            return df
        except Exception as e:
            print("Error renaming dataframe columns: " + e.__str__())
            raise

    @staticmethod
    def removing_missing_values(df):
        """Removing missing values"""
        try:
            df.dropna(subset=['item_name'], inplace=True)
            return df
        except Exception as e:
            print("Error removing missing values: " + e.__str__())
            raise

    @staticmethod
    def removing_anomalies(df):
        """Removing anomalies"""
        try:
            df = df[df['item_name'] != 'Discount']
            return df
        except Exception as e:
            print("Error removing anomalies: " + e.__str__())
            raise

    @staticmethod
    def delete_stopwords(df):
        """Delete stopwords"""
        try:
            stop = stopwords.words('english')
            df['item_name_transformed'] = df['item_name'].apply(
                lambda x: ' '.join([word for word in x.split() if word not in stop])
            )
            return df
        except Exception as e:
            print("Error to delete stopwords: " + e.__str__())
            raise

    @staticmethod
    def text_lemmatize(df):
        """Lemmatize words (reduce the term by transforming it into its lemma)"""
        try:
            spacy_nlp = spacy.load('en_core_web_lg')
            df['item_name_transformed'] = df['item_name_transformed'].apply(
                lambda row: " ".join([w.lemma_ for w in spacy_nlp(row)])
            )
            return df
        except Exception as e:
            print("Error to lemmatize words: " + e.__str__())
            raise

    @staticmethod
    def data_capitalization(df):
        """Convert text to lowercase"""
        try:
            df['item_name_transformed'] = df['item_name_transformed'].str.lower()
            return df
        except Exception as e:
            print("Error to text capitalization: " + e.__str__())
            raise

    @staticmethod
    def convert_df_parquet(df):
        """Convert dataframe file to parquet"""
        try:
            import os
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            df.to_parquet(
                os.path.join(BASE_DIR, "data", "purchase-order-data-2012-2015-.parquet"),
                compression='brotli'
            )
            return df
        except Exception as e:
            print("Error converting dataframe to Parquet file: " + e.__str__())
            raise

    def text_query_preprocessing(self, text_query):
        """Preprocessing the text query"""
        try:
            # Convert search text to dataframe temporarily
            dict_query = {'item_name': [text_query]}
            df_query = pd.DataFrame(dict_query)

            # Reuse existing preprocessing methods
            df_query = self.removing_anomalies(df_query)
            df_query = self.delete_stopwords(df_query)
            df_query = self.text_lemmatize(df_query)
            df_query = self.data_capitalization(df_query)

            # Return preprocessed text
            result_text_query_preprocessing = df_query.item_name_transformed.loc[0]
            return result_text_query_preprocessing
        except Exception as e:
            print("Error preprocessing the text query: " + e.__str__())

    def data_preprocessing(self):
        """Preprocessing dataframe"""
        try:
            self.df_purchase_order = self.df_purchase_order.head(100)
            self.set_dataframe_index()
            self.df_purchase_order = self.delete_columns(self.df_purchase_order)
            self.df_purchase_order = self.rename_columns_name(self.df_purchase_order)
            self.df_purchase_order = self.removing_missing_values(self.df_purchase_order)
            self.df_purchase_order = self.removing_anomalies(self.df_purchase_order)
            self.df_purchase_order = self.delete_stopwords(self.df_purchase_order)
            self.df_purchase_order = self.text_lemmatize(self.df_purchase_order)
            self.df_purchase_order = self.data_capitalization(self.df_purchase_order)
            self.convert_df_parquet(self.df_purchase_order)
            # print(self.df_purchase_order)
        except Exception as e:
            print("Error preprocessing the dataframe: " + e.__str__())


# Run
purchaseOrderPreprocessingDomain = PurchaseOrderPreprocessingDomain()
purchaseOrderPreprocessingDomain.data_preprocessing()