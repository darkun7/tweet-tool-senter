import os
from flask import Flask, render_template, request, json, send_from_directory
import backend as ml

app = Flask(__name__, template_folder='template')

@app.context_processor
def utility_processor():
    def drive_image(ids):
        return 'https://drive.google.com/uc?export=view&id='+ids
    return dict(drive_image=drive_image)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'img/favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/")
def landingPage():
    return render_template("index.html")

@app.route("/about")
def aboutPage():
    return render_template("about.html")

@app.route("/team/<username>")
def teamPage(username):
    try:
        SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
        json_url = os.path.join(SITE_ROOT, "static/dev", username+".json")
        data = json.load(open(json_url))
        return render_template("team.html", data=data)
    except (FileNotFoundError, IOError):
        return "Not Found"

@app.route("/analyze",methods=['POST'])
def analyze():
    res = ml.main(request)
    return render_template('result.html', res=res)
if __name__=="__main__":
    app.run(debug=True)