import os
import pickle
from sentence_transformers import SentenceTransformer
from domain.purchases.PurchaseOrderRepository import PurchaseOrderRepository

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKL_PATH = os.path.join(BASE_DIR, "data", "purchase-order-data-2012-2015-.pkl")


class PurchaseOrderWordEmbeddingsDomain:

    @staticmethod
    def create_word_embeddings_file():
        """Create serialized file"""
        try:
            purchaseOrderRepository = PurchaseOrderRepository()
            df = purchaseOrderRepository.load_purchase_order_parquet()
            df.reset_index(drop=True, inplace=True)

            model = SentenceTransformer('all-mpnet-base-v2')
            corpus_embeddings = model.encode(
                df["item_name_transformed"].tolist(),
                convert_to_tensor=True
            )

            with open(PKL_PATH, "wb") as fOut:
                pickle.dump(
                    {'embeddings': corpus_embeddings},
                    fOut,
                    protocol=pickle.HIGHEST_PROTOCOL
                )
            print("Embeddings file created successfully!")

        except Exception as e:
            print("Error to generate serialize embeddings file: " + e.__str__())
            raise

    @staticmethod
    def load_word_embeddings_file_transformed():
        """Load serialized Embeddings file"""
        try:
            with open(PKL_PATH, "rb") as fIn:
                stored_data = pickle.load(fIn)
                corpus_embeddings = stored_data['embeddings']
                return corpus_embeddings
        except Exception as e:
            print("Error loading serialized Embeddings file: " + e.__str__())
            raise

    @staticmethod
    def transform_text_query_word_embeddings(query_embeddings):
        """Transforms search query text into Word Embeddings"""
        try:
            model = SentenceTransformer('all-mpnet-base-v2')
            query_embeddings = model.encode(
                query_embeddings,
                convert_to_tensor=True
            )
            return query_embeddings
        except Exception as e:
            print("Error transforming query text into Word Embeddings: " + e.__str__())
            raise


# Run
# purchaseOrderWordEmbeddingsDomain = PurchaseOrderWordEmbeddingsDomain()
# purchaseOrderWordEmbeddingsDomain.create_word_embeddings_file()