#include "neuralnetwork.h"
#include <QDebug>
#include <cmath>
#include <algorithm>
#include <numeric>
const double EPSILON = 1e-8;
const double LEARNING_RATE = 0.001;
const double DROPOUT_RATE = 0.3;
const double L2_LAMBDA = 0.0001;
const double BATCH_NORM_MOMENTUM = 0.9;
NeuralNetwork::NeuralNetwork(int inputNum, int hidden1Num, int hidden2Num, int outputNum)
    : m_inputSize(inputNum)
    , m_hidden1Size(hidden1Num)
    , m_hidden2Size(hidden2Num)
    , m_outputSize(outputNum)
    , m_learningRate(LEARNING_RATE)
    , m_lastLoss(0.0)
    , m_dropoutRate(DROPOUT_RATE)
    , m_l2Lambda(L2_LAMBDA)
{
    m_input.resize(m_inputSize);
    m_hidden1.resize(m_hidden1Size);
    m_hidden2.resize(m_hidden2Size);
    m_outputs.resize(m_outputSize);
    m_hidden1Drop.resize(m_hidden1Size);
    m_hidden2Drop.resize(m_hidden2Size);
    m_weightsInputHidden1.resize(m_inputSize);
    for (int i = 0; i < m_inputSize; i++) {
        m_weightsInputHidden1[i].resize(m_hidden1Size);
    }

    m_weightsHidden1Hidden2.resize(m_hidden1Size);
    for (int i = 0; i < m_hidden1Size; i++) {
        m_weightsHidden1Hidden2[i].resize(m_hidden2Size);
    }

    m_weightsHidden2Output.resize(m_hidden2Size);
    for (int i = 0; i < m_hidden2Size; i++) {
        m_weightsHidden2Output[i].resize(m_outputSize);
    }
    heInitialize(m_weightsInputHidden1, m_inputSize);
    heInitialize(m_weightsHidden1Hidden2, m_hidden1Size);
    heInitialize(m_weightsHidden2Output, m_hidden2Size);
    m_biasHidden1.resize(m_hidden1Size);
    m_biasHidden2.resize(m_hidden2Size);
    m_biasOutput.resize(m_outputSize);

    initializeBiases(m_biasHidden1, m_hidden1Size);
    initializeBiases(m_biasHidden2, m_hidden2Size);
    initializeBiases(m_biasOutput, m_outputSize);
    m_gammaHidden1.resize(m_hidden1Size);
    m_betaHidden1.resize(m_hidden1Size);
    m_gammaHidden2.resize(m_hidden2Size);
    m_betaHidden2.resize(m_hidden2Size);

    m_runningMeanHidden1.resize(m_hidden1Size);
    m_runningVarHidden1.resize(m_hidden1Size);
    m_runningMeanHidden2.resize(m_hidden2Size);
    m_runningVarHidden2.resize(m_hidden2Size);
    for (int i = 0; i < m_hidden1Size; i++) {
        m_gammaHidden1[i] = 1.0;
        m_betaHidden1[i] = 0.0;
        m_runningMeanHidden1[i] = 0.0;
        m_runningVarHidden1[i] = 1.0;
    }

    for (int i = 0; i < m_hidden2Size; i++) {
        m_gammaHidden2[i] = 1.0;
        m_betaHidden2[i] = 0.0;
        m_runningMeanHidden2[i] = 0.0;
        m_runningVarHidden2[i] = 1.0;
    }
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::heInitialize(QVector<QVector<double>> &weights, int fanIn)
{
    double stddev = std::sqrt(2.0 / fanIn);

    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[i].size(); j++) {
            weights[i][j] = (m_random.generateDouble() * 2.0 - 1.0) * stddev;
        }
    }
}

void NeuralNetwork::initializeBiases(QVector<double> &biases, int size)
{
    for (int i = 0; i < size; i++) {
        biases[i] = 0.01;
    }
}

double NeuralNetwork::relu(double x)
{
    return x > 0 ? x : 0;
}

double NeuralNetwork::reluDerivative(double x)
{
    return x > 0 ? 1 : 0;
}

double NeuralNetwork::softmax(const QVector<double> &z, int index)
{
    double maxVal = *std::max_element(z.begin(), z.end());
    double sum = 0.0;

    for (int i = 0; i < z.size(); i++) {
        sum += std::exp(z[i] - maxVal);
    }

    return std::exp(z[index] - maxVal) / (sum + EPSILON);
}
void NeuralNetwork::batchNormalize(QVector<double> &layer,
                                   const QVector<double> &gamma,
                                   const QVector<double> &beta,
                                   QVector<double> &runningMean,
                                   QVector<double> &runningVar,
                                   bool training)
{
    int size = layer.size();
    if (training) {
        double mean = 0.0;
        for (int i = 0; i < size; i++) {
            mean += layer[i];
        }
        mean /= size;

        double var = 0.0;
        for (int i = 0; i < size; i++) {
            var += std::pow(layer[i] - mean, 2);
        }
        var = var / size + EPSILON;
        for (int i = 0; i < size; i++) {
            runningMean[i] = BATCH_NORM_MOMENTUM * runningMean[i] +
                             (1.0 - BATCH_NORM_MOMENTUM) * mean;
            runningVar[i] = BATCH_NORM_MOMENTUM * runningVar[i] +
                            (1.0 - BATCH_NORM_MOMENTUM) * var;
        }
        for (int i = 0; i < size; i++) {
            layer[i] = (layer[i] - mean) / std::sqrt(var);
            layer[i] = gamma[i] * layer[i] + beta[i];
        }
    } else {
        for (int i = 0; i < size; i++) {
            layer[i] = gamma[i] * (layer[i] - runningMean[i]) /
                           std::sqrt(runningVar[i] + EPSILON) + beta[i];
        }
    }
}
void NeuralNetwork::forward(const QVector<double> &input, bool training)
{
    m_input = input;
    for (int j = 0; j < m_hidden1Size; j++) {
        double sum = m_biasHidden1[j];
        for (int i = 0; i < m_inputSize; i++) {
            sum += m_input[i] * m_weightsInputHidden1[i][j];
        }
        m_hidden1[j] = sum;
    }
    batchNormalize(m_hidden1, m_gammaHidden1, m_betaHidden1,
                   m_runningMeanHidden1, m_runningVarHidden1, training);

    // ReLU активация
    for (int j = 0; j < m_hidden1Size; j++) {
        m_hidden1[j] = relu(m_hidden1[j]);
    }

    // Dropout (только при обучении)
    if (training) {
        for (int j = 0; j < m_hidden1Size; j++) {
            if (m_random.generateDouble() < m_dropoutRate) {
                m_hidden1Drop[j] = 0;
            } else {
                m_hidden1Drop[j] = m_hidden1[j] / (1.0 - m_dropoutRate);
            }
        }
    } else {
        m_hidden1Drop = m_hidden1;
    }

    // Слой 2: скрытый1 -> скрытый2
    for (int k = 0; k < m_hidden2Size; k++) {
        double sum = m_biasHidden2[k];
        for (int j = 0; j < m_hidden1Size; j++) {
            sum += m_hidden1Drop[j] * m_weightsHidden1Hidden2[j][k];
        }
        m_hidden2[k] = sum;
    }

    // Batch normalization для скрытого слоя 2
    batchNormalize(m_hidden2, m_gammaHidden2, m_betaHidden2,
                   m_runningMeanHidden2, m_runningVarHidden2, training);

    // ReLU активация
    for (int k = 0; k < m_hidden2Size; k++) {
        m_hidden2[k] = relu(m_hidden2[k]);
    }

    // Dropout (только при обучении)
    if (training) {
        for (int k = 0; k < m_hidden2Size; k++) {
            if (m_random.generateDouble() < m_dropoutRate) {
                m_hidden2Drop[k] = 0;
            } else {
                m_hidden2Drop[k] = m_hidden2[k] / (1.0 - m_dropoutRate);
            }
        }
    } else {
        m_hidden2Drop = m_hidden2;
    }

    // Выходной слой: скрытый2 -> выход
    for (int l = 0; l < m_outputSize; l++) {
        double sum = m_biasOutput[l];
        for (int k = 0; k < m_hidden2Size; k++) {
            sum += m_hidden2Drop[k] * m_weightsHidden2Output[k][l];
        }
        m_outputs[l] = sum;
    }

    // Softmax для выходного слоя
    QVector<double> outputsSoftmax(m_outputSize);
    for (int l = 0; l < m_outputSize; l++) {
        outputsSoftmax[l] = softmax(m_outputs, l);
    }
    m_outputs = outputsSoftmax;
}

// Исправленная функция backward (без неиспользуемого параметра input)
void NeuralNetwork::backward(int target)
{
    // Вычисляем градиенты для выходного слоя
    // Для softmax + cross-entropy: dL/dz = y_pred - y_true
    QVector<double> outputGradients(m_outputSize);
    for (int i = 0; i < m_outputSize; i++) {
        if (i == target) {
            outputGradients[i] = m_outputs[i] - 1.0;
        } else {
            outputGradients[i] = m_outputs[i];
        }
    }

    // Градиенты для весов скрытый2->выход
    QVector<QVector<double>> dWeightsHidden2Output(m_hidden2Size);
    QVector<double> dBiasOutput(m_outputSize, 0.0);

    for (int k = 0; k < m_hidden2Size; k++) {
        dWeightsHidden2Output[k].resize(m_outputSize);
        for (int l = 0; l < m_outputSize; l++) {
            dWeightsHidden2Output[k][l] = m_hidden2Drop[k] * outputGradients[l];
        }
    }

    for (int l = 0; l < m_outputSize; l++) {
        dBiasOutput[l] = outputGradients[l];
    }

    // Градиенты для скрытого слоя 2
    QVector<double> hidden2Gradients(m_hidden2Size, 0.0);
    for (int k = 0; k < m_hidden2Size; k++) {
        double sum = 0.0;
        for (int l = 0; l < m_outputSize; l++) {
            sum += m_weightsHidden2Output[k][l] * outputGradients[l];
        }
        hidden2Gradients[k] = sum * reluDerivative(m_hidden2[k]);
    }

    // Градиенты для весов скрытый1->скрытый2
    QVector<QVector<double>> dWeightsHidden1Hidden2(m_hidden1Size);
    QVector<double> dBiasHidden2(m_hidden2Size, 0.0);

    for (int j = 0; j < m_hidden1Size; j++) {
        dWeightsHidden1Hidden2[j].resize(m_hidden2Size);
        for (int k = 0; k < m_hidden2Size; k++) {
            dWeightsHidden1Hidden2[j][k] = m_hidden1Drop[j] * hidden2Gradients[k];
        }
    }

    for (int k = 0; k < m_hidden2Size; k++) {
        dBiasHidden2[k] = hidden2Gradients[k];
    }

    // Градиенты для скрытого слоя 1
    QVector<double> hidden1Gradients(m_hidden1Size, 0.0);
    for (int j = 0; j < m_hidden1Size; j++) {
        double sum = 0.0;
        for (int k = 0; k < m_hidden2Size; k++) {
            sum += m_weightsHidden1Hidden2[j][k] * hidden2Gradients[k];
        }
        hidden1Gradients[j] = sum * reluDerivative(m_hidden1[j]);
    }

    // Градиенты для весов вход->скрытый1
    QVector<QVector<double>> dWeightsInputHidden1(m_inputSize);
    QVector<double> dBiasHidden1(m_hidden1Size, 0.0);

    for (int i = 0; i < m_inputSize; i++) {
        dWeightsInputHidden1[i].resize(m_hidden1Size);
        for (int j = 0; j < m_hidden1Size; j++) {
            dWeightsInputHidden1[i][j] = m_input[i] * hidden1Gradients[j];
        }
    }

    for (int j = 0; j < m_hidden1Size; j++) {
        dBiasHidden1[j] = hidden1Gradients[j];
    }

    // Обновляем веса с L2 регуляризацией
    for (int i = 0; i < m_inputSize; i++) {
        for (int j = 0; j < m_hidden1Size; j++) {
            m_weightsInputHidden1[i][j] -= m_learningRate *
                                           (dWeightsInputHidden1[i][j] + m_l2Lambda * m_weightsInputHidden1[i][j]);
        }
    }

    for (int j = 0; j < m_hidden1Size; j++) {
        for (int k = 0; k < m_hidden2Size; k++) {
            m_weightsHidden1Hidden2[j][k] -= m_learningRate *
                                             (dWeightsHidden1Hidden2[j][k] + m_l2Lambda * m_weightsHidden1Hidden2[j][k]);
        }
    }

    for (int k = 0; k < m_hidden2Size; k++) {
        for (int l = 0; l < m_outputSize; l++) {
            m_weightsHidden2Output[k][l] -= m_learningRate *
                                            (dWeightsHidden2Output[k][l] + m_l2Lambda * m_weightsHidden2Output[k][l]);
        }
    }

    // Обновляем смещения
    for (int j = 0; j < m_hidden1Size; j++) {
        m_biasHidden1[j] -= m_learningRate * dBiasHidden1[j];
    }

    for (int k = 0; k < m_hidden2Size; k++) {
        m_biasHidden2[k] -= m_learningRate * dBiasHidden2[k];
    }

    for (int l = 0; l < m_outputSize; l++) {
        m_biasOutput[l] -= m_learningRate * dBiasOutput[l];
    }

    // Вычисляем потерю (кросс-энтропия)
    m_lastLoss = -std::log(m_outputs[target] + EPSILON);
}

void NeuralNetwork::train(const QVector<double> &input, int target)
{
    forward(input, true);
    backward(target);
}

int NeuralNetwork::predict(const QVector<double> &input)
{
    forward(input, false);

    // Находим максимальную вероятность
    int bestIndex = 0;
    double maxOutput = m_outputs[0];

    for (int i = 1; i < m_outputSize; i++) {
        if (m_outputs[i] > maxOutput) {
            maxOutput = m_outputs[i];
            bestIndex = i;
        }
    }

    return bestIndex;
}

bool NeuralNetwork::trainOnDataset(const QVector<TrainingData> &dataset, int epochs)
{
    if (dataset.isEmpty()) {
        qDebug() << "Пустой датасет для обучения";
        return false;
    }

    // Перемешиваем данные перед каждой эпохой
    QVector<TrainingData> shuffled = dataset;

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Перемешивание
        std::random_shuffle(shuffled.begin(), shuffled.end());

        double totalLoss = 0.0;
        int correct = 0;

        for (const TrainingData &data : shuffled) {
            // Находим целевую цифру
            int targetDigit = 0;
            double maxTarget = data.target[0];
            for (int i = 1; i < data.target.size(); i++) {
                if (data.target[i] > maxTarget) {
                    maxTarget = data.target[i];
                    targetDigit = i;
                }
            }

            // Обучаем на примере
            train(data.input, targetDigit);
            totalLoss += m_lastLoss;

            // Проверяем, правильно ли распознано
            int predicted = predict(data.input);
            if (predicted == targetDigit) {
                correct++;
            }
        }

        double averageLoss = totalLoss / shuffled.size();
        double accuracy = static_cast<double>(correct) / shuffled.size() * 100.0;

        qDebug() << QString("Эпоха %1/%2 | Потеря: %3 | Точность: %4%")
                        .arg(epoch + 1).arg(epochs)
                        .arg(averageLoss, 0, 'f', 4)
                        .arg(accuracy, 0, 'f', 2);
    }

    return true;
}

bool NeuralNetwork::saveModel(const QString &filename)
{
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly)) {
        qDebug() << "Не удалось открыть файл для записи:" << filename;
        return false;
    }

    QDataStream out(&file);
    out.setVersion(QDataStream::Qt_5_15);

    // Записываем архитектуру
    out << m_inputSize << m_hidden1Size << m_hidden2Size << m_outputSize;

    // Записываем веса
    for (int i = 0; i < m_inputSize; i++) {
        for (int j = 0; j < m_hidden1Size; j++) {
            out << m_weightsInputHidden1[i][j];
        }
    }

    for (int i = 0; i < m_hidden1Size; i++) {
        for (int j = 0; j < m_hidden2Size; j++) {
            out << m_weightsHidden1Hidden2[i][j];
        }
    }

    for (int i = 0; i < m_hidden2Size; i++) {
        for (int j = 0; j < m_outputSize; j++) {
            out << m_weightsHidden2Output[i][j];
        }
    }

    // Записываем смещения
    for (int i = 0; i < m_hidden1Size; i++) {
        out << m_biasHidden1[i];
    }

    for (int i = 0; i < m_hidden2Size; i++) {
        out << m_biasHidden2[i];
    }

    for (int i = 0; i < m_outputSize; i++) {
        out << m_biasOutput[i];
    }

    // Записываем batch normalization параметры
    for (int i = 0; i < m_hidden1Size; i++) {
        out << m_gammaHidden1[i] << m_betaHidden1[i];
        out << m_runningMeanHidden1[i] << m_runningVarHidden1[i];
    }

    for (int i = 0; i < m_hidden2Size; i++) {
        out << m_gammaHidden2[i] << m_betaHidden2[i];
        out << m_runningMeanHidden2[i] << m_runningVarHidden2[i];
    }

    file.close();
    return true;
}

bool NeuralNetwork::loadModel(const QString &filename)
{
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "Не удалось открыть файл для чтения:" << filename;
        return false;
    }

    QDataStream in(&file);
    in.setVersion(QDataStream::Qt_5_15);

    int inputSize, hidden1Size, hidden2Size, outputSize;
    in >> inputSize >> hidden1Size >> hidden2Size >> outputSize;

    if (inputSize != m_inputSize || hidden1Size != m_hidden1Size ||
        hidden2Size != m_hidden2Size || outputSize != m_outputSize) {
        qDebug() << "Архитектура модели не совпадает";
        file.close();
        return false;
    }

    // Читаем веса
    for (int i = 0; i < m_inputSize; i++) {
        for (int j = 0; j < m_hidden1Size; j++) {
            in >> m_weightsInputHidden1[i][j];
        }
    }

    for (int i = 0; i < m_hidden1Size; i++) {
        for (int j = 0; j < m_hidden2Size; j++) {
            in >> m_weightsHidden1Hidden2[i][j];
        }
    }

    for (int i = 0; i < m_hidden2Size; i++) {
        for (int j = 0; j < m_outputSize; j++) {
            in >> m_weightsHidden2Output[i][j];
        }
    }

    // Читаем смещения
    for (int i = 0; i < m_hidden1Size; i++) {
        in >> m_biasHidden1[i];
    }

    for (int i = 0; i < m_hidden2Size; i++) {
        in >> m_biasHidden2[i];
    }

    for (int i = 0; i < m_outputSize; i++) {
        in >> m_biasOutput[i];
    }

    // Читаем batch normalization параметры
    for (int i = 0; i < m_hidden1Size; i++) {
        in >> m_gammaHidden1[i] >> m_betaHidden1[i];
        in >> m_runningMeanHidden1[i] >> m_runningVarHidden1[i];
    }

    for (int i = 0; i < m_hidden2Size; i++) {
        in >> m_gammaHidden2[i] >> m_betaHidden2[i];
        in >> m_runningMeanHidden2[i] >> m_runningVarHidden2[i];
    }

    file.close();
    return true;
}

bool NeuralNetwork::saveCorrections(const QString &filename, const QVector<TrainingData> &corrections)
{
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }

    QDataStream out(&file);
    out.setVersion(QDataStream::Qt_5_15);

    out << corrections.size();
    for (const TrainingData &data : corrections) {
        out << data.input << data.target << data.label;
    }

    file.close();
    return true;
}

bool NeuralNetwork::loadCorrections(const QString &filename, QVector<TrainingData> &corrections)
{
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }

    QDataStream in(&file);
    in.setVersion(QDataStream::Qt_5_15);

    int size;
    in >> size;

    corrections.clear();
    for (int i = 0; i < size; i++) {
        TrainingData data;
        in >> data.input >> data.target >> data.label;
        corrections.append(data);
    }

    file.close();
    return true;
}

void NeuralNetwork::testNetwork()
{
    // Создаем тестовый вход
    QVector<double> testInput(m_inputSize);
    for (int i = 0; i < m_inputSize; i++) {
        testInput[i] = m_random.generateDouble();
    }

    forward(testInput, false);

    qDebug() << "Тест сети:";
    qDebug() << "Выходные нейроны:";
    for (int i = 0; i < m_outputSize; i++) {
        qDebug() << QString("  %1: %2").arg(i).arg(m_outputs[i], 0, 'f', 4);
    }

    double sum = 0.0;
    for (int i = 0; i < m_outputSize; i++) {
        sum += m_outputs[i];
    }
    qDebug() << QString("Сумма вероятностей: %1").arg(sum, 0, 'f', 4);
}

QString NeuralNetwork::getModelInfo() const
{
    return QString("Архитектура: %1->%2->%3->%4 | LR: %5 | Dropout: %6 | L2: %7")
        .arg(m_inputSize).arg(m_hidden1Size).arg(m_hidden2Size).arg(m_outputSize)
        .arg(m_learningRate).arg(m_dropoutRate).arg(m_l2Lambda);
}
