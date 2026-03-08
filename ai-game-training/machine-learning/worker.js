importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;

let _labels = []
let _model = null
async function loadModelAndLabels() {
    await tf.ready()
    _labels = await (await fetch(LABELS_PATH)).json()
    _model = await tf.loadGraphModel(MODEL_PATH)

    const dummyInput = tf.ones(_model.inputs[0].shape)
    await _model.executeAsync(dummyInput)
    tf.dispose(dummyInput)

    postMessage({ type: 'model-loaded' })
}
loadModelAndLabels()


self.onmessage = async ({ data }) => {
    if (data.type !== 'predict') return

    postMessage({
        type: 'prediction',
        x: 400,
        y: 400,
        score: 0
    });


};

console.log('🧠 YOLOv5n Web Worker initialized');
