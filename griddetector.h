#ifndef GRIDDETECTOR_H
#define GRIDDETECTOR_H
#include <QImage>
#include <QVector>
#include <QRect>
#include "commontypes.h"
class GridDetector {
public:
    GridDetector();
    GridDetectionResult detectGrid(const QImage &image);
    bool isCellEmpty(const QImage &cellImage) const;
    QImage removeBlueLines(const QImage &image) const;
private:
    GridDetectionResult detectBlueGrid(const QImage &image);
    GridDetectionResult detectBlackGrid(const QImage &image);
    GridDetectionResult detectSimpleGrid(const QImage &image);
    bool isBluePixel(QRgb pixel) const;
    bool isDarkPixel(QRgb pixel) const;
    QImage createBinaryImage(const QImage &image, bool useBlueDetection) const;
    QVector<int> findLines(const QImage &binaryImage, bool horizontal) const;
    QImage dilate(const QImage &image, int kernelSize = 3) const;
};
#endif
