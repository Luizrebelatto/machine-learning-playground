import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = null
const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
}

function normalize(value, min, max){
    return (value - min) / ((max - min) || 1)
}

export function makeContext(products, users){
    const ages = users.map(user => user.age);
    const prices = products.map(product => product.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(products.map(product => product.color))]
    const categories = [...new Set(products.map(product => product.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        })
    )

    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        })
    )
    
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(purchase => {
            ageSums[purchase.name] = (ageSums[purchase.name] || 0) + user.age
            ageCounts[purchase.name] = (ageCounts[purchase.name] || 0) + 1
        })
    });
    
    const productAvgAgeNormalized = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ? 
                ageSums[product.name] / ageCounts[product.name] : midAge;
            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )
    
    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        minAge, maxAge,
        minPrice, maxPrice,
        productAvgAgeNormalized,
        numCategories: categories.length,
        numColors: colors.length,
        dimensions: 2 + categories.length + colors.length
    } 
}

const oneHotWeighted = (index, length, weight) => tf.oneHot(index, length).cast('float32').mul(weight);

function encodeProduct(product, context){
    const price = tf.tensor1d([normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price]);
    const age = tf.tensor1d([(context.productAvgAgeNormalized[product.name] ?? 0.5) * WEIGHTS.age]);
    const category = oneHotWeighted(context.categoriesIndex[product.category], context.numCategories, WEIGHTS.category);
    const color = oneHotWeighted(context.colorsIndex[product.color], context.numColors, WEIGHTS.color);

    return tf.concat1d([price, age, category, color ])
}

function encodeUser(user, context){
    if(user.purchases.length){
        return tf.stack(
            user.purchases.map(product => encodeProduct(product, context))
        ).mean(0)
        .reshape([
            1, context.dimensions
        ])
    }
    return tf.concat1d([
            tf.zeros([1]), // price is ignored
            tf.tensor1d([
                normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age
            ]),
            tf.zeros([context.numCategories]), // ignore category
            tf.zeros([context.numColors]), // ignore colors
        ]
    ).reshape([1, context.dimensions])
}

function createTrainingData(context){
    const inputs = []
    const labels = []
    context.users.filter(user => user.purchases.length)
    .forEach(user => {
        const userVector = encodeUser(user, context).dataSync();
        context.products.forEach(product => {
            const productVector = encodeProduct(product, context).dataSync()
            const label = user.purchases.some(
                purchase => purchase.name === product.name ? 1 : 0
            )
            inputs.push([...userVector, ...productVector])
            labels.push(label)
        })
    })
    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimension: context.dimensions * 2 // userVector + ProductVector
    }
}

async function configureNeuralNetAndTrain(trainData){
    const model = tf.sequential();

    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimension],
            units: 128,
            activation: 'relu'
        })
    )

    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )

    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )

    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }))
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })
    
    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }  
        }
    })
    return model
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)
    const products = await (await fetch("../../data/products.json")).json();
    
    const context = makeContext(products, users)
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync()
        }
    })

    _globalCtx = context

    const trainData = createTrainingData(context)
    _model = await configureNeuralNetAndTrain(trainData)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}
function recommend(user) {
    if(!_model) return;
    
    const context = _globalCtx
    const userVector = encodeUser(user, context).dataSync()
    
    const inputs = context.productVectors.map(({ vector }) => {
        return [...userVector, ...vector]
    })

    const inputTensor = tf.tensor2d(inputs);
    const predictions = _model.predict(inputTensor);
    const scores = predictions.dataSync();

    const recommendations = context.productVectors.map((item, index) => {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index],

        }
    })
    const sortedItems = recommendations.sort((a, b) => b.score - a.score)

    postMessage({ type: workerEvents.recommend, user, recommendations: sortedItems });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
