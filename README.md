# GDSC_FER_PROJECT
Overview : This project is created for GDSC AI/ML Domain Induction task, which asked the students to create a model which can indentiy facial expressions of an user and to do that I used Opencv library of python and also for deep leanirng CNN model I have used tensorflow.

Breif Description of Function: The approach which I am using here is that it first of all takes the driectory containg the images the I wrote a function to convert in into a dataframme containg two columns one of them being label containg expression and the other one containg the image path, then using another function cextract_feature images and extracted and the the model is trained on these images , the model consists of various layers like Convolutional layer max pooliing layer, dropout layer and flatten and oviously Dense layer.
then I convert this model into a json file which I uses in another file and using it I recognise face expressions.

DEMO: If you want you can see the demo of my project here :
https://drive.google.com/file/d/161mCl9pwLmbbBeIfuBZnKKayEWGTPiv-/view?usp=drive_link

In case you want to run it yourself:

Step 1: Run "pip install -r requirement.txt" on your laptop to install all reqiurement 

Step 2: Run Hope.py then FinalRunner.py, it will take a lot of time if you do not have GPU as I didn't.
