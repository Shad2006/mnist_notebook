#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <QVector>
#include <QRandomGenerator>
#include <QFile>
#include <QDataStream>
#include "commontypes.h"

class NeuralNetwork {
public:
    NeuralNetwork(int inputNum = 784, int hidden1Num = 256,
                  int hidden2Num = 128, int outputNum = 10);

    ~NeuralNetwork();
    void train(const QVector<double> &input, int target);
    int predict(const QVector<double> &input);
    bool trainOnDataset(const QVector<TrainingData> &dataset, int epochs = 10);
    bool saveModel(const QString &filename);
    bool loadModel(const QString &filename);
    bool saveCorrections(const QString &filename, const QVector<TrainingData> &corrections);
    bool loadCorrections(const QString &filename, QVector<TrainingData> &corrections);
    double getLoss() const { return m_lastLoss; }
    QVector<double> getOutputs() const { return m_outputs; }
    QString getModelInfo() const;
    void testNetwork();

private:
    int m_inputSize;
    int m_hidden1Size;
    int m_hidden2Size;
    int m_outputSize;
    QVector<double> m_input;
    QVector<double> m_hidden1;
    QVector<double> m_hidden2;
    QVector<double> m_outputs;
    QVector<double> m_hidden1Drop;
    QVector<double> m_hidden2Drop;
    QVector<QVector<double>> m_weightsInputHidden1;
    QVector<QVector<double>> m_weightsHidden1Hidden2;
    QVector<QVector<double>> m_weightsHidden2Output;
    QVector<double> m_biasHidden1;
    QVector<double> m_biasHidden2;
    QVector<double> m_biasOutput;
    QVector<double> m_gammaHidden1;
    QVector<double> m_betaHidden1;
    QVector<double> m_gammaHidden2;
    QVector<double> m_betaHidden2;
    QVector<double> m_runningMeanHidden1;
    QVector<double> m_runningVarHidden1;
    QVector<double> m_runningMeanHidden2;
    QVector<double> m_runningVarHidden2;
    double m_learningRate;
    double m_lastLoss;
    double m_dropoutRate;
    double m_l2Lambda;
    QRandomGenerator m_random;
    static double relu(double x);
    static double reluDerivative(double x);
    static double softmax(const QVector<double> &z, int index);
    void forward(const QVector<double> &input, bool training = true);
    void backward(int target);
    void batchNormalize(QVector<double> &layer,
                        const QVector<double> &gamma,
                        const QVector<double> &beta,
                        QVector<double> &runningMean,
                        QVector<double> &runningVar,
                        bool training);
    void heInitialize(QVector<QVector<double>> &weights, int fanIn);
    void initializeBiases(QVector<double> &biases, int size);
};
#endif
