import sys
from flask import Flask, request, render_template
from src.pipelines.prdict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=request.form.get('reading_score'),
                writing_score=request.form.get('writing_score')
            )

            inputs_df = data.get_data_as_dataframe()

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(inputs_df)
            # print("="*200, '\n', results,"\n","="*200)
            return render_template('home.html', results=results[0])
        except Exception as e:
            raise CustomException(e, sys)
    else:
        return render_template('home.html')
    
if __name__ == "__main__":
    application.run(host='0.0.0.0')