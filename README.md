# Web-Scraper-and-Rent-Price-Prediction-with-Images

To measure the influence of appartment images, the information of thousends of exposés in Berlin were scraped inclusive the images and the groundplan.

> **Requirements** <br>
> pip install pdf2image

Workflow:
1. Scrape the data with *1_Immoscout_WebScraper.py* by including the exposéID and the amount of cities <br>
2. Pick the images and save them to a corresponding folder with *2_Immoscout_Image_Selection.py*, e.g. the kitchen and bathroom images <br>
3. Select the intersection dataset which has all, a kitchen and a bathroom image with *3_Immoscout_Data_Selection.py* <br>
4. Preprocess the dataset with *4_Immoscout_Data_Preprocessing.py* <br>
5. Preprocess the images with *5_Immoscout_Intersection_Dataset_Image_Preprocessing_listing.py* <br>
6. Create a model which predicts the rent/sqm with *6_Immoscout_Modeling.py* <br>
7. Analyse the dataset and visualize the results with some plots by *7_Immoscout_Plots.py*
