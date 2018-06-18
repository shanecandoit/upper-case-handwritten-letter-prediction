'''
from flask import *
from datetime import datetime

#from flask.ext.heroku import Heroku
#ExtDeprecationWarning: Importing flask.ext.heroku is deprecated, use flask_heroku instead.
from flask_heroku import Heroku

app = Flask(__name__)

@app.route('/')
def homepage():
    the_time = datetime.now()#.strftime("%A, %d %b %Y %l:%M %p")

    return """
    <h1>Hello heroku</h1>
    <p>It is currently {time}.</p>

    <img src="http://loremflickr.com/600/400">
    """.format(time=the_time)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
'''


from flask import *
from PIL import Image
import PIL.ImageOps

import numpy as np

import base64
import os, sys
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='')

# http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # limit upload size

import random
app.secret_key = str(random.random())[2:]

import time
def ms():
    return str(int(time.time()*1000))

SMALL = (28,28)

model = None

# work around for flask bug
# https://github.com/keras-team/keras/issues/2397
flask_workaround = True
if flask_workaround:
    import tensorflow as tf
    graph = None

def load_model():
    """ load model from disk. hard coded path. """
    global model
    if flask_workaround:
        global graph
    # load from disk and overwrite existing global model object
    from keras.models import model_from_json
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #print('load_model model',model)
    # load weights into new model
    model.load_weights("model.h5")
    
    # flask work around
    # https://github.com/keras-team/keras/issues/2397
    if flask_workaround:
        graph = tf.get_default_graph()
        #print('load_model graph',graph)
    #print("Loaded model from disk")# load from disk and overwrite existing global model object
    print('load_model model',model)
    return model

## PREDICT
def str2d_250(ary):
    res=''
    for r in ary:
        for c in r:
            ch=' '
            if c<250:
                ch='*'
            res+=ch
        res+='\n'
    return res

from keras.preprocessing import image
from PIL import Image
from PIL import ImageFilter
def model_predict(img_path, model):
    img = Image.open( img_path )
    
    # from 3 channels to 1
    img_grey = img.convert("L")
    
    ## black-on-white to white-on-black
    ## try no blur
    #g_blur = img_grey.filter(ImageFilter.GaussianBlur(5))
    target=28,28
    #small = g_blur.resize(target)
    small = img_grey.resize(target)
    #small_inv = (255-small)
    small_inv = PIL.ImageOps.invert(small)
    smallfile = img_path.replace('.png','-small.png')
    small_inv.save( smallfile )
    #print 'small'
    
    ## np array from image
    # Preprocessing the image
    #x = image.img_to_array(small).astype('float32')
    # csv-to-image uses uint8
    #x = small_inv.img_to_array(small).astype('uint8')
    image_array = np.asarray( small_inv.getdata() )
    image_array = image_array.reshape(28, 28, 1).astype('float32')
    #print('image_array',image_array.shape)
    #print('image_array',image_array)
    x = image_array
    
    ## test the np array
    #test_image = Image.fromarray(image_array.astype('uint8'))
    #test_image_p = img_path.replace('.png','-np.png')
    #test_image.save( test_image_p )
    
    #print x.shape
    print( str2d_250(x))
    
    # invert colors
    # we trained on inverted colors
    #print('invert')
    #x = (255-x)
    #print 'x'
    #print x.shape
    #print str2d_0(x)
    
    # ValueError: Error when checking input: expected conv2d_1_input to have 4 dimensions, but got array with shape (1, 28, 28)
    samp = np.expand_dims( x, axis=0) # to put the [3] into a [4]
    #print 'samp.shape',samp.shape
    #print('model_predict model',model)

    # flask work around
    # https://github.com/keras-team/keras/issues/2397
    if flask_workaround:
        global graph
        print('graph',graph)
        with graph.as_default():
            #print('graph',graph)
            preds = model.predict(samp)
            #return preds
            scores = letter_scores( preds )
            print('score',scores)
            return scores
    else:
        preds = model.predict(samp)
        #return preds
        scores = letter_scores( preds )
        print('score',scores)
        return scores



# delete old files
def delete_old():
    import os
    try:
        files = os.listdir('uploads/')
        if len(files) > 50:
            old5 = os.listdir('uploads/')[:5]
            for old in old5:
                os.remove('uploads/'+old)
    except FileNotFoundError:
        pass

@app.route('/')
def index():
    #here=os.path.abspath(__file__)
    #print('here',here)
    #near=os.listdir('.')
    #print('near',near)
    #above=os.listdir('../')
    #print('above',above)
    global model
    delete_old()
    if not model:
        model = load_model()
        print('loaded model',model)
    return app.send_static_file('draw.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload/', methods=['GET', 'POST'])
def upload_file():
    delete_old()
    # check model
    global model
    global graph
    if not model:
        model = load_model()
    if flask_workaround and not graph:
        graph = tf.get_default_graph()
    #global model # doesnt write to global model, so dont put global here?
    if request.method == 'POST':
        #print(dir(request))
        if request.form['text']:
            tx = request.form['text']
            #print('text',tx)
            #here=os.path.abspath(__file__)
            #print('upload here',here)
            filename=ms()+'.png'
            save_b64(tx, filename)
            #resize('uploads/'+filename)
            if model:
                print('upload has model')
            else:
                print('upload has no model')
            result = model_predict( 'uploads/'+filename, model)
            # {'A' : 1.0 } # cant serial 1.0, make it a string then
            result = { k:str(v) for k,v in result.items() }
            #print('result', result)

            #print('text',tx)
            #return redirect('/uploads/'+filename)
            #js_res = {'link': '/uploads/'+filename , 'score': jsonify(result) }
            js_res = {'link': '/uploads/'+filename , 'score': result }
            #print('js_res',js_res)
            #return '/uploads/'+filename + '|' + str(result)
            jsond = jsonify( js_res )
            #print('jsond',jsond)
            return jsond

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload Image</title>
    <h1>Upload Image</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=text name=text>
         <input type=submit value=Upload>
    </form>
    '''

# process upload png
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    #return send_from_directory(app.config['UPLOAD_FOLDER'],
    #                           filename)
    print('filename',filename)
    print('text',request.form['text'])
    if request.method == 'POST':
        save_b64(text, filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})


# upload image as b64 text
def save_b64(img_data,filename='img.png'):
    with open('uploads/'+filename, 'wb') as fh:
        fh.write(base64.b64decode(img_data))

def letter_scores(scorelist):
    """ take an array of 26 scores
    
    scores2letter({ chr(i+ord('A')) : x for i,x in enumerate([1,0,0.5,]) if x})
    = {'A': 1, 'C': 0.5}
    
    """
    #letter_scores={ chr(i+ord('A')):x for i,x in enumerate( scores ) if x>0.1}
    let_score = { chr(i+ord('A')):x for i,x in enumerate( scorelist[0] ) if x>0.01}
    return let_score

# predict
# https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
'''
not using this yet

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)
'''

# main
if __name__ == '__main__':
    load_model()
    app.run(debug=True,host='127.0.0.1')

'''
according to:
https://stackoverflow.com/questions/43822458/loading-a-huge-keras-model-into-a-flask-app

this error
ValueError: Tensor Tensor("dense_2/Softmax:0", shape=(?, 26), dtype=float32) is not an element of this graph.

this page:
https://github.com/keras-team/keras/issues/2397

says its a webservice fail.
- try: avital commented on Oct 19, 2016
- works

'''


