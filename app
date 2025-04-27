from flask import Flask, render_template

# Create the Flask app
app = Flask(__name__)

# Define a simple home route
@app.route('/')
def home():
    return "Hello, this is your production monitoring dashboard!"