import os
from flask import Flask, render_template, request, jsonify, json

app = Flask(__name__, template_folder='template')

@app.context_processor
def utility_processor():
    def drive_image(ids):
        return 'https://drive.google.com/uc?export=view&id='+ids
    return dict(drive_image=drive_image)

@app.route("/")
def landingPage():
    return render_template("index.html")

@app.route("/about")
def teamPage():
    return render_template("about.html")

@app.route("/team/<username>")
def aboutPage(username):
    try:
        SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
        json_url = os.path.join(SITE_ROOT, "static/dev", username+".json")
        data = json.load(open(json_url))
        return render_template("team.html", data=data)
    except (FileNotFoundError, IOError):
        return "Not Found"
    

if __name__=="__main__":
    app.run(debug=True)