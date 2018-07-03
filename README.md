
# Upper case handwritten letter prediction

A flask app, deployed to Heroku.
**[GO See it](https://hidden-abilites-ocr-demo.herokuapp.com/)**

## Why?

I wanted to show friends and family what machine learning can do.

There are lots of good resources on classifying MNIST.

Here is one that deploys to heroku:
[github.com/sugyan/tensorflow-mnist](https://github.com/sugyan/tensorflow-mnist)

a video:
[MLPaint: the Real-Time Handwritten Digit Recognizer
](https://www.youtube.com/watch?v=WGdLCXDiDSo)

## My twist

Rather than copy-paste and learn only a little.
I decided to find a dataset, smoosh it into a csv, train the model, and deploy to the web.

## How
1. get data 
   - [nist.gov/itl/iad/image-group/emnist-dataset](https://www.nist.gov/itl/iad/image-group/emnist-dataset)
2. turn into a csv
3. train model on csv
   - [kaggle.com/ashishguptajiit/cnn-using-keras-accuracy-after-10-epochs-98-89](https://www.kaggle.com/ashishguptajiit/cnn-using-keras-accuracy-after-10-epochs-98-89)
   - 600 mb csv file (not checked in, bake your own ;)
4. commmit model to repo
5. write code to load model and predict on images POSTed
5. deploy 

## Paint in Browser

Borrowed from here:
[draw-on-html5-canvas-using-a-mouse](https://stackoverflow.com/questions/2368784/draw-on-html5-canvas-using-a-mouse)

![Screenshot](https://raw.githubusercontent.com/shanecandoit/upper-case-handwritten-letter-prediction/master/preview.png)