#ifndef COMMONTYPES_H
#define COMMONTYPES_H

#include <QVector>
#include <QString>
#include <QImage>
#include <QRect>

struct TrainingData {
    QVector<double> input;
    QVector<double> target;
    QString label;
};

struct RecognizedCell {
    QString character;
    double confidence;
    QRect boundingRect;
    QImage cellImage;
    QImage processedImage;
    QVector<double> inputVector;
    bool corrected;
    QString userCorrectedChar;
    int row;
    int col;
};

struct GridDetectionResult {
    QImage processedImage;
    QVector<QRect> cells;
    QVector<QImage> cellImages;
    int gridRows;
    int gridCols;
};

struct ManualGridSettings {
    int rows;
    int cols;
    int margin;
};

#endif
