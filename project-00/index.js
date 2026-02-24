import tf from '@tensorflow/tfjs-node';

// const people = [
// 	{ name: "Luiz", age: 30, color: "blue", localization: "cachoeirinha" },
// 	{ name: "Gabriel", age: 22, color: "yellow", localization: "gravatai" },
// 	{ name: "fred", age: 21, color: "pink", localization: "porto alegre" }
// ]

// const categories = [
// 	"Luiz", "premium",
// 	"Gabriel", "medium",
// 	"Carlos", "basic"
// ]

const peopleTensorNormalized = [
	[1, 1, 0, 0, 1, 0, 0], // Luiz
	[0.11, 0, 1, 0, 0, 1, 0], // Gabriel
	[0, 0, 0, 1, 0, 0, 1], // Fred
]

const labels = ["premium", "medium", "basic"]
const tensorLabels = [
	[1, 0, 0], // Luiz - premium
	[0, 1, 0], // Gabriel - medium 
	[0, 0, 1] // Carlos - basic
]

const inputXs = tf.tensor2d(peopleTensorNormalized)
const outputYs = tf.tensor2d(tensorLabels)

inputXs.print()
outputYs.print()