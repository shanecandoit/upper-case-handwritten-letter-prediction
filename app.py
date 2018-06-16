#from flask import Flask, render_template, request
from Flask import *
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

