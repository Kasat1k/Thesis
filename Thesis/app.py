from flask import Flask, render_template, request, redirect, url_for
import os
from detector_module import run_edge_detection

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        alpha = float(request.form.get('alpha', 0.5))
        sigma = float(request.form.get('sigma', 1.0))

        if image:
            path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(path)

            recommended_alpha, recommended_sigma = run_edge_detection(path, alpha, sigma)

            return render_template('index.html',
                                   rl_path='static/result_rl.png',
                                   cf_path='static/result_cf.png',
                                   cv_path='static/result_cv.png',
                                   uploaded=True,
                                   recommended_alpha=recommended_alpha,
                                   recommended_sigma=recommended_sigma)

    return render_template('index.html', uploaded=False)
if __name__ == "__main__":
    app.run(debug=True)