******This is a work in progress*****

***Requirements:
Flask==2.3.2
pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.0
scikit-multilearn==0.3.0
requests==2.32.0
joblib==1.3.2


Manga genre classifier based on getting data from a DB that receives info from Mangadex

1. First, get the data with Mangadex
- Be sure to remove redundant data; I did not do that 
- Instead, check if an en or jp title exists

2. Classify it into different categories of columns
- Need to change genre to true or false ( 1 or 0) 
- Visual classification of book cover is another thing:
  - A neural network (like a Convolutional Neural Network, CNN) is used to extract visual patterns from the cover.
  - Leave it for now, might do in future, it's just that if I test with new data, it will unnecessarily download the image data locally

3. Train a model using a validation method (supervised)
- Could try k-fold cross-validation
- Could dabble in unsupervised learning and use nearest neighbour for prediction
- Could also do reinforcement learning
- Need to find a way to translate titles and use that to also classify the genre

4. Compare which one is better (do only when I make the others)

5. Test it out with new manga
  - So make a website that takes in a link from manga dex of any manga out there, and use that to predict its value and give an accuracy result when u compare with the actual one
    - So the left side is predicted, and the right side is actual
