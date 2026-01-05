***Don't need to train the model, trained model is already in the project files.
***You can delete the index_old.html file in templates folder if you want.
***You can also delete the merged_crop_yield_data_fields.csv if you want.

----------------------------------------------------------------------------------------------
RUN THE PROJECT COMMANDS : 
    first open the cmd in the current folder location then run below commands one by one:

        pip install -r requirements.txt

        python app.py

----------------------------------------------------------------------------------------------
MODEL TRAINING COMMANDS : 
    for model training run below commands one by one:

        python train_recommendation_model.py

        python train_yield_model.py

----------------------------------------------------------------------------------------------
Important Information:
Model accuracy graphs are in static folder in image format.
All datasets which are required for the model training are in dataset folder.
All models after model training is in the models folder.