#ifndef DATASETLOADER_H
#define DATASETLOADER_H
#include <QVector>
#include <QString>
#include <QImage>
#include "commontypes.h"
class DatasetLoader {
public:
    DatasetLoader();
    bool loadCustomDataset(const QString &datasetPath);
    bool loadOldDataset(const QString &datasetPath);
    QVector<TrainingData> getTrainingData() const { return m_trainingData; }
    int getSampleCount() const { return m_trainingData.size(); }
private:
    QVector<TrainingData> m_trainingData;

    QVector<double> imageToVector(const QImage &image);
    QImage preprocessImage(const QImage &image);
};
#endif
