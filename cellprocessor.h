#ifndef CELLPROCESSOR_H
#define CELLPROCESSOR_H
#include <QImage>
#include <QVector>
#include "commontypes.h"
class CellProcessor {
public:
    CellProcessor();

    QVector<double> preprocessCell(const QImage &cellImage);
    QImage getProcessedImage(const QImage &cellImage);
private:
    QImage removeNoise(const QImage &image) const;
    QImage binarize(const QImage &image) const;
};
#endif
