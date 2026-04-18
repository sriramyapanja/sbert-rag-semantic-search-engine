import json
from flask import Flask, Response, request
from flasgger import Swagger
from domain.purchases.PurchaseOrderPreprocessingDomain import PurchaseOrderPreprocessingDomain
from domain.purchases.PurchaseOrderWordEmbeddingsDomain import PurchaseOrderWordEmbeddingsDomain
from domain.purchases.PurchaseOrderDomain import PurchaseOrderDomain

app = Flask(__name__)
Swagger(app)


class SemanticSearchApi:
    """Semantic Search Api"""

    @staticmethod
    @app.route('/')
    def get():
        return "Semantic Text Search API using S-BERT."

    @staticmethod
    @app.route('/purchaseorder/preprocessing', methods=['POST'])
    def pre_processing_purchase_order():
        """Endpoint for data preprocessing before ML model execution
        ---
        tags:
          - Data preprocessing
        responses:
          201:
            description: Success preprocessing file
          500:
            description: Failure preprocessing file
        """
        try:
            purchaseOrderPreprocessingDomain = PurchaseOrderPreprocessingDomain()
            purchaseOrderPreprocessingDomain.data_preprocessing()
            return Response(
                '{"result": "Success to preprocessed the Dataset."}',
                mimetype='application/json'
            ), 201
        except Exception as e:
            print(e)
            return Response(
                '{"result": "Failure to preprocessed the Dataset."}',
                mimetype='application/json'
            ), 500

    @staticmethod
    @app.route('/purchaseorder/semanticsearch', methods=['POST'])
    def semantic_search():
        """Endpoint to semantic search
        ---
        tags:
          - Semantic Search
        parameters:
          - name: body
            in: body
            required: true
        responses:
          200:
            description: Success to find records.
          500:
            description: Failure to search.
        """
        try:
            json_query = json.loads(request.data)
            purchaseOrderDomain = PurchaseOrderDomain()
            json_result = purchaseOrderDomain.semantic_search(json_query['sentence_query'])
            json_result = json_result.to_json(orient="records")
            return Response(json_result, mimetype='application/json'), 200
        except Exception as e:
            print(e)
            return Response(
                '{"result": "Failure to search."}',
                mimetype='application/json'
            ), 500


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")