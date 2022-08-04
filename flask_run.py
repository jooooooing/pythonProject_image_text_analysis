import numpy as np
import tensorflow as tf
import time
from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask('First App')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

@app.route("/compare", methods = ['POST'])
def comparePhoto():
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    def upload_file():
        if request.method == 'POST':
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
                filename = "uploads/" + str(int(time.time())) + ".jpg"
                file.save(filename)
                input_text = filename
            return filename
    def create_graph():
        """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
        # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
        with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')


    result = {}
    imagePath = upload_file()
    modelFullPath = '/tmp/output_graph.pb'                                      # 읽어들일 graph 파일 경로
    labelsFullPath = '/tmp/output_labels.txt'

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return result

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    create_graph()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%d %s (score = %.5f)' % (node_id, human_string, score))
            save_string = ""
            if node_id == 0:
                save_string = "shitzu"
            elif node_id == 1:
                save_string = "frenchbulldog"
            else:
                save_string = "maltese"

            result[save_string] = score
        answer = labels[top_k[0]]
    ##
    result_convert = {k:float(v) for k,v in result.items()}
    return result_convert

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
