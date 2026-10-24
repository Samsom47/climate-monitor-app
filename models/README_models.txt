Models Folder
=============

1. landcover_classifier.joblib
   - Real trained RandomForest model
   - Classes: Vegetation, Water, Sand/Soil, Buildings/Roads, Mixed
   - Input: 9 simple color & texture features extracted from the image

2. label_map.joblib
   - Maps class numbers to human-readable names

3. feature_names.joblib
   - Names of the features the model expects

How it works:
- Image → extract color averages, std, green/blue ratios → model predicts class
- This is a real machine learning model (not just if-else rules)

Vegetation green-ratio tool still uses direct pixel calculation
because it is more accurate and easy for students to understand.
