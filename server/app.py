import os
import time
import threading


from flask import Flask, request, jsonify, redirect, url_for, send_file, after_this_request
from flask_cors import CORS
from model_api import get_model_api



DELETE_DELAY_TIME = 60 * 1 * 0.1
FILE_NAME = 'DEEP_SIGHT_FILE'
UPLOAD_FOLDER = '.\\uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'avi', 'mp4'])


sem = threading.Semaphore()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 * 40
CORS(app)

model_image_api, model_video_api = get_model_api()


# Default route
@app.route('/')
def index():
    return "Index API"


# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


# Error Handler
@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


# API route
@app.route('/upload', methods=['POST'])
def api():

    # Measure time
    start = time.time()

    # Check whether file has been uplaoded
    if FILE_NAME not in request.files or request.files[FILE_NAME] is None:
        print('No file part')
        return redirect(request.url)

    file = request.files[FILE_NAME]

    # Check whether file has allowed extension
    if not allowed_file(file.filename):
        print('File not supported')
        return redirect(request.url)

    # Get type of file
    type, ext = file.mimetype.split('/')

    sem.acquire()

    if type == 'image':

        out = model_image_api(file)
        # Print total time
        stop = time.time() - start
        print('Total request time : {0: .3f}s'.format(stop))
        response = jsonify(out[0])

    elif type == 'video':

        out_path, new_path = model_video_api(file, app.config['UPLOAD_FOLDER'])
        # Print total time
        stop = time.time() - start
        print('Total request time : {0: .3f}s'.format(stop))
        response = send_file(out_path, mimetype=file.mimetype, attachment_filename=file.filename, as_attachment=True)

        # Delete files after serving
        delete_file(new_path)
        delete_file(out_path)
   
    else:

        # File type not supported
        print('File type not supported')
        return redirect(request.url)

    sem.release()

    return response


def delete_file(path, delay=DELETE_DELAY_TIME):
     # Create and start thread 
     del_thread = threading.Thread(target=delay_delete, args=(path, delay))
     del_thread.start()


def delay_delete(path, delay):
    # Sleep and then try to remove file
    time.sleep(delay)
    os.remove(path)

    return


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)

