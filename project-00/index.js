import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs){
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // output layer with 3 units
    // softmax activation function to get probabilities for each class
    model.add(tf.layers.dense({ units: 3, activation: 'softmax'}));
    // Optimizer Adam(Adaptive moment estimation)
    // ajusta os pesos de forma eficiente e inteligente
    // aprende com historico de erros e acertos

    // loss: compara os scores de cada categoria com a resposta correta
    // mais distante da previsao, maior o erro
    // classificacao de imagens, recomendacoes, categorizar usuarios, 1 opcao para os retornos
    model.compile({ optimizer: 'adam', loss: "categoricalCrossentropy", metrics: ['accuracy']})

    // epochs: ira passar 100 vezes pela base de dados
    // verbose: nao ira mostrar logs
    // shuffle: embaralha os dados a cada epoca, para evitar que o modelo aprenda a ordem dos dados, evitar um algoritmo viciado
    await model.fit(inputXs, outputYs,{
        verbose: 0,
        epochs: 100,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch: ${epoch}, Loss: ${logs.loss}, Accuracy: ${logs.acc}`)
                
            }
        }
    })
    // salva em disco

    return model;

}

async function predict(model, person){
    // transforma o array em tensor
    const tfInput = tf.tensor2d(person)
    // faz a previsao, output sera um vetor de 3 probabilidades, uma para cada categoria
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({
        prob, index
    }))
}

const peopleTensorNormalized = [
	[1, 1, 0, 0, 1, 0, 0], // Luiz
	[0.11, 0, 1, 0, 0, 1, 0], // Gabriel
	[0, 0, 0, 1, 0, 0, 1], // Fred
]

const labels = ["Premium", "Medium", "Basic"]
const tensorLabels = [
	[1, 0, 0], // Luiz - premium
	[0, 1, 0], // Gabriel - medium 
	[0, 0, 1] // Fred - basic
]

const inputXs = tf.tensor2d(peopleTensorNormalized)
const outputYs = tf.tensor2d(tensorLabels)

const model = await trainModel(inputXs, outputYs)

// pessoas que nao estava no treinamento, para testar o modelo
const person = { name: "Pedro", age: 24, color: "pink", localization: "porto alegre" };
const personTensorNormalized = [[0.1, 0, 0, 1, 0, 0, 1]]

const predictions = await predict(model, personTensorNormalized)
const result = predictions.sort((a, b) => b.prob - a.prob).map(p => `${labels[p.index]} (${(p.prob * 100).toFixed(2)}%)`).join("\n")

console.log(result)