import os
from dotenv import load_dotenv
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

def load_configuration():
    """Loads configuration settings from environment variables."""
    load_dotenv()
    return (
        os.getenv('PredictionEndpoint'),
        os.getenv('PredictionKey'),
        os.getenv('ProjectID'),
        os.getenv('ModelName')
    )

def authenticate_client(prediction_endpoint, prediction_key):
    """Authenticates a client for the Custom Vision prediction API."""
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    return CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

def classify_images(prediction_client, project_id, model_name, image_dir):
    """Classifies images in a directory using the provided model."""
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        with open(image_path, "rb") as image_data:
            results = prediction_client.classify_image(project_id, model_name, image_data)
            for prediction in results.predictions:
                if prediction.probability > 0.5:
                    print(f"{image_name}: {prediction.tag_name} ({prediction.probability:.4f})")

def main():
    """Main entry point of the program."""
    try:
        prediction_endpoint, prediction_key, project_id, model_name = load_configuration()
        prediction_client = authenticate_client(prediction_endpoint, prediction_key)
        image_dir = r'D:\Internship1\Task4\testing_code\test_image'  # Replace with your image directory
        classify_images(prediction_client, project_id, model_name, image_dir)
    except Exception as ex:
        print(f"An error occurred: {ex}")

if __name__ == "__main__":
    main()
