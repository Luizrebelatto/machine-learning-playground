import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

export function makeContext(catalog, users){
    const ages = users.map(user => user.age);
    const prices = catalog.map(product => product.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(catalog.map(product => product.color))]
    const categories = [...new Set(catalog.map(product => product.category))]

    const colorIndex = Object.entries(
        colors.map((color, index) => {
            return [color, index]
        })
    )

    const categoriesIndex = Object.entries(
        categories.map((category, index) => {
            return [category, index]
        })
    )
    
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)
    const catalog = await (await fetch("../../data/products.json")).json();
    
    const context = makeContext(catalog, users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
