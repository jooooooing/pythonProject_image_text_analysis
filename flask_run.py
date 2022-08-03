from flask_restful import reqparse
import numpy as np
import tensorflow as tf

from flask import Flask
from flask_cors import CORS

app = Flask('First App')
CORS(app)

@app.route("/compare")
def comparePhoto():
    def create_graph():
        """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
        # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
        with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    result = {}

    parser = reqparse.RequestParser()
    parser.add_argument('path', required=True)
    args = parser.parse_args()
    input_text = args['path']

    #imagePath = '/var/www/html/show/files/' + review_text
    imagePath = input_text
    modelFullPath = '/tmp/output_graph.pb'                                      # 읽어들일 graph 파일 경로
    labelsFullPath = '/tmp/output_labels.txt'

    if not tf.gfile.Exists(imagePath):
        tf.compat.logging.fatal('File does not exist %s', imagePath)
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
