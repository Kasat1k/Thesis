from flask import Flask, render_template, request, session
import os
from werkzeug.utils import secure_filename
from detector_module import run_edge_detection, auto_select_parameters

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    image_path = session.get('last_image_path')
    recommended_alpha = session.get('recommended_alpha', 0.5)
    recommended_sigma = session.get('recommended_sigma', 1.0)

    if request.method == 'POST':
        image = request.files.get('image')
        alpha = request.form.get('alpha', type=float)
        sigma = request.form.get('sigma', type=float)

        if image:
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            session['last_image_path'] = image_path

            recommended_alpha, recommended_sigma, _ = auto_select_parameters(image_path)
            session['recommended_alpha'] = recommended_alpha
            session['recommended_sigma'] = recommended_sigma

            alpha = recommended_alpha
            sigma = recommended_sigma

        if image_path:
            run_edge_detection(image_path, alpha, sigma)

            return render_template('index.html',
                                   original='static/uploaded.png',
                                   rl='static/result_rl.png',
                                   cf='static/result_cf.png',
                                   canny='static/result_cv.png',
                                   uploaded=True,
                                   alpha=session['recommended_alpha'],
                                   sigma=session['recommended_sigma'])

    return render_template('index.html',
                           uploaded=bool(image_path),
                           original='static/uploaded.png' if image_path else None,
                           rl=None, cf=None, canny=None,
                           alpha=recommended_alpha,
                           sigma=recommended_sigma)

if __name__ == '__main__':
    app.run(debug=True)
