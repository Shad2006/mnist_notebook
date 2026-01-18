#include "cellprocessor.h"
#include <QPainter>
#include <QDebug>
#include <algorithm>
CellProcessor::CellProcessor() {
}
QVector<double> CellProcessor::preprocessCell(const QImage &cellImage)
{
    QImage processed = getProcessedImage(cellImage);
    QVector<double> result(784);
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            QRgb pixel = processed.pixel(x, y);
            int gray = qGray(pixel);
            double value = (255.0 - gray) / 255.0;
            result[y * 28 + x] = value;
        }
    }
    return result;
}
QImage CellProcessor::getProcessedImage(const QImage &cellImage)
{
    if (cellImage.isNull()) {
        return QImage(28, 28, QImage::Format_Grayscale8);
    }
    QImage denoised = removeNoise(cellImage);
    QImage binary = binarize(denoised);
    QImage scaled = binary.scaled(20, 20, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QImage result(28, 28, QImage::Format_Grayscale8);
    result.fill(255);
    QPainter painter(&result);
    int xOffset = (28 - scaled.width()) / 2;
    int yOffset = (28 - scaled.height()) / 2;
    painter.drawImage(xOffset, yOffset, scaled);
    painter.end();
    return result;
}
QImage CellProcessor::removeNoise(const QImage &image) const
{
    QImage result = image;
    if (image.width() < 3 || image.height() < 3) {
        return result;
    }
    for (int y = 1; y < image.height() - 1; y++) {
        for (int x = 1; x < image.width() - 1; x++) {
            QVector<int> neighbors;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    neighbors.append(qGray(image.pixel(x + dx, y + dy)));
                }
            }

            std::sort(neighbors.begin(), neighbors.end());
            int median = neighbors[4];
            result.setPixel(x, y, qRgb(median, median, median));
        }
    }
    return result;
}
QImage CellProcessor::binarize(const QImage &image) const
{
    QImage result(image.size(), QImage::Format_Grayscale8);
    int totalGray = 0;
    int pixelCount = image.width() * image.height();
    for (int y = 0; y < image.height(); y++) {
        for (int x = 0; x < image.width(); x++) {
            totalGray += qGray(image.pixel(x, y));
        }
    }
    int threshold = totalGray / pixelCount;
    for (int y = 0; y < image.height(); y++) {
        for (int x = 0; x < image.width(); x++) {
            int gray = qGray(image.pixel(x, y));
            int binary = (gray < threshold) ? 0 : 255;
            result.setPixel(x, y, qRgb(binary, binary, binary));
        }
    }
    return result;
}
