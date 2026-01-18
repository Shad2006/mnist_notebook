#include "griddetector.h"
#include <QPainter>
#include <QDebug>
#include <algorithm>
#include <QColor>
#include <cmath>
GridDetector::GridDetector() {
}
GridDetectionResult GridDetector::detectGrid(const QImage &image) {
    GridDetectionResult result;
    result = detectBlueGrid(image);
    if (result.cells.size() >= 4) {
        qDebug() << "Обнаружена голубая сетка:" << result.cells.size() << "клеток";
        return result;
    }
    result = detectBlackGrid(image);
    if (result.cells.size() >= 4) {
        qDebug() << "Обнаружена черная сетка:" << result.cells.size() << "клеток";
        return result;
    }
    result = detectSimpleGrid(image);
    qDebug() << "Используется простая сетка:" << result.cells.size() << "клеток";

    return result;
}
GridDetectionResult GridDetector::detectBlueGrid(const QImage &image) {
    GridDetectionResult result;
    if (image.isNull()) {
        return result;
    }
    QImage binaryImage = createBinaryImage(image, true);
    QVector<int> horizontalLines = findLines(binaryImage, true);
    QVector<int> verticalLines = findLines(binaryImage, false);
    if (horizontalLines.size() < 2 || verticalLines.size() < 2) {
        return result;
    }
    QImage processedImage = image.copy();
    QPainter painter(&processedImage);
    painter.setPen(QPen(Qt::cyan, 2));
    std::sort(horizontalLines.begin(), horizontalLines.end());
    std::sort(verticalLines.begin(), verticalLines.end());
    for (int y : horizontalLines) {
        painter.drawLine(0, y, processedImage.width(), y);
    }
    for (int x : verticalLines) {
        painter.drawLine(x, 0, x, processedImage.height());
    }
    for (int i = 0; i < horizontalLines.size() - 1; i++) {
        for (int j = 0; j < verticalLines.size() - 1; j++) {
            int top = horizontalLines[i];
            int bottom = horizontalLines[i + 1];
            int left = verticalLines[j];
            int right = verticalLines[j + 1];
            if (bottom - top > 10 && right - left > 10) {
                QRect cellRect(left, top, right - left, bottom - top);
                QRect innerRect = cellRect.adjusted(2, 2, -2, -2);
                if (innerRect.isValid()) {
                    QImage cellImage = image.copy(innerRect);
                    if (!isCellEmpty(cellImage)) {
                        result.cells.append(innerRect);
                        result.cellImages.append(cellImage);
                        painter.setPen(QPen(Qt::green, 1));
                        painter.drawRect(innerRect);
                    }
                }
            }
        }
    }
    painter.end();
    result.processedImage = processedImage;
    return result;
}
GridDetectionResult GridDetector::detectBlackGrid(const QImage &image) {
    GridDetectionResult result;
    if (image.isNull()) {
        return result;
    }
    QImage binaryImage = createBinaryImage(image, false);
    binaryImage = dilate(binaryImage, 2);
    QVector<int> horizontalLines;
    QVector<int> verticalLines;
    for (int y = 0; y < binaryImage.height(); y++) {
        int linePixels = 0;
        for (int x = 0; x < binaryImage.width(); x++) {
            if (binaryImage.pixelIndex(x, y) == 1) {
                linePixels++;
            }
        }
        if (linePixels > binaryImage.width() * 0.15) {
            horizontalLines.append(y);
        }
    }
    for (int x = 0; x < binaryImage.width(); x++) {
        int linePixels = 0;
        for (int y = 0; y < binaryImage.height(); y++) {
            if (binaryImage.pixelIndex(x, y) == 1) {
                linePixels++;
            }
        }
        if (linePixels > binaryImage.height() * 0.15) {
            verticalLines.append(x);
        }
    }
    if (horizontalLines.size() < 2 || verticalLines.size() < 2) {
        return result;
    }
    auto mergeLines = [](QVector<int> lines, int minGap = 3) {
        QVector<int> merged;
        for (int line : lines) {
            if (merged.isEmpty() || qAbs(line - merged.last()) > minGap) {
                merged.append(line);
            }
        }
        return merged;
    };

    horizontalLines = mergeLines(horizontalLines);
    verticalLines = mergeLines(verticalLines);
    QImage processedImage = image.copy();
    QPainter painter(&processedImage);
    painter.setPen(QPen(Qt::red, 2));
    std::sort(horizontalLines.begin(), horizontalLines.end());
    std::sort(verticalLines.begin(), verticalLines.end());
    for (int y : horizontalLines) {
        painter.drawLine(0, y, processedImage.width(), y);
    }

    for (int x : verticalLines) {
        painter.drawLine(x, 0, x, processedImage.height());
    }
    for (int i = 0; i < horizontalLines.size() - 1; i++) {
        for (int j = 0; j < verticalLines.size() - 1; j++) {
            int top = horizontalLines[i];
            int bottom = horizontalLines[i + 1];
            int left = verticalLines[j];
            int right = verticalLines[j + 1];
            if (bottom - top > 10 && right - left > 10) {
                QRect cellRect(left, top, right - left, bottom - top);
                QRect innerRect = cellRect.adjusted(2, 2, -2, -2);
                if (innerRect.isValid()) {
                    QImage cellImage = image.copy(innerRect);
                    if (!isCellEmpty(cellImage)) {
                        result.cells.append(innerRect);
                        result.cellImages.append(cellImage);
                        painter.setPen(QPen(Qt::yellow, 1));
                        painter.drawRect(innerRect);
                    }
                }
            }
        }
    }
    painter.end();
    result.processedImage = processedImage;
    return result;
}
QImage GridDetector::removeBlueLines(const QImage &image) const
{
    if (image.isNull()) {
        return QImage();
    }

    QImage result = image.copy();

    for (int y = 0; y < image.height(); y++) {
        for (int x = 0; x < image.width(); x++) {
            QRgb pixel = image.pixel(x, y);
            if (isBluePixel(pixel)) {
                result.setPixel(x, y, qRgb(255, 255, 255));
            } else {
                int gray = qGray(pixel);
                if (gray > 100) {
                    result.setPixel(x, y, qRgb(255, 255, 255));
                }
            }
        }
    }
    return result;
}

GridDetectionResult GridDetector::detectSimpleGrid(const QImage &image) {
    GridDetectionResult result;

    if (image.isNull()) {
        return result;
    }
    QImage processedImage = image.copy();
    QPainter painter(&processedImage);
    painter.setPen(QPen(Qt::blue, 2));

    int width = image.width();
    int height = image.height();
    int rows = 5;
    int cols = 5;
    int cellWidth = width / cols;
    int cellHeight = height / rows;
    for (int i = 0; i <= rows; i++) {
        int y = i * cellHeight;
        painter.drawLine(0, y, width, y);
    }
    for (int i = 0; i <= cols; i++) {
        int x = i * cellWidth;
        painter.drawLine(x, 0, x, height);
    }
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int left = col * cellWidth;
            int top = row * cellHeight;
            int right = left + cellWidth;
            int bottom = top + cellHeight;

            QRect cellRect(left, top, cellWidth, cellHeight);
            QRect innerRect = cellRect.adjusted(5, 5, -5, -5);

            if (innerRect.isValid()) {
                QImage cellImage = image.copy(innerRect);

                if (!isCellEmpty(cellImage)) {
                    result.cells.append(innerRect);
                    result.cellImages.append(cellImage);

                    painter.setPen(QPen(Qt::green, 1));
                    painter.drawRect(innerRect);
                }
            }
        }
    }

    painter.end();
    result.processedImage = processedImage;
    return result;
}

bool GridDetector::isCellEmpty(const QImage &cellImage) const {
    if (cellImage.isNull()) return true;

    int darkPixels = 0;
    int totalPixels = cellImage.width() * cellImage.height();

    for (int y = 0; y < cellImage.height(); y++) {
        for (int x = 0; x < cellImage.width(); x++) {
            QRgb pixel = cellImage.pixel(x, y);
            int gray = qGray(pixel);

            if (gray < 200) {
                darkPixels++;
            }
            if (darkPixels > totalPixels * 0.03) {
                return false;
            }
        }
    }

    return true;
}

QImage GridDetector::createBinaryImage(const QImage &image, bool useBlueDetection) const {
    QImage binaryImage(image.size(), QImage::Format_Mono);
    binaryImage.fill(0);

    for (int y = 0; y < image.height(); y++) {
        for (int x = 0; x < image.width(); x++) {
            QRgb pixel = image.pixel(x, y);
            bool isLinePixel = false;

            if (useBlueDetection) {
                isLinePixel = isBluePixel(pixel);
            } else {
                isLinePixel = isDarkPixel(pixel);
            }

            if (isLinePixel) {
                binaryImage.setPixel(x, y, 1);
            }
        }
    }

    return binaryImage;
}

bool GridDetector::isBluePixel(QRgb pixel) const {
    int r = qRed(pixel);
    int g = qGreen(pixel);
    int b = qBlue(pixel);
    bool isCyan = (b > 150 && g > 150 && r < 100);
    bool blueDominant = (b > r + 50 && b > g + 30);
    bool lightBlue = (b > 200 && g > 180 && r < 150);

    return isCyan || blueDominant || lightBlue;
}

bool GridDetector::isDarkPixel(QRgb pixel) const {
    int r = qRed(pixel);
    int g = qGreen(pixel);
    int b = qBlue(pixel);
    int brightness = (r + g + b) / 3;
    return brightness < 150;
}

QVector<int> GridDetector::findLines(const QImage &binaryImage, bool horizontal) const {
    QVector<int> lines;
    if (horizontal) {
        for (int y = 0; y < binaryImage.height(); y++) {
            int linePixels = 0;

            for (int x = 0; x < binaryImage.width(); x++) {
                if (binaryImage.pixelIndex(x, y) == 1) {
                    linePixels++;
                }
            }
            if (linePixels > binaryImage.width() * 0.25) { // 25%
                lines.append(y);
            }
        }
    } else {
        for (int x = 0; x < binaryImage.width(); x++) {
            int linePixels = 0;

            for (int y = 0; y < binaryImage.height(); y++) {
                if (binaryImage.pixelIndex(x, y) == 1) {
                    linePixels++;
                }
            }

            if (linePixels > binaryImage.height() * 0.25) { // 25%
                lines.append(x);
            }
        }
    }
    QVector<int> mergedLines;
    int minGap = 5;

    for (int line : lines) {
        if (mergedLines.isEmpty() || qAbs(line - mergedLines.last()) > minGap) {
            mergedLines.append(line);
        }
    }

    return mergedLines;
}

QImage GridDetector::dilate(const QImage &image, int kernelSize) const {
    QImage result = image;
    int half = kernelSize / 2;

    for (int y = half; y < image.height() - half; y++) {
        for (int x = half; x < image.width() - half; x++) {
            bool found = false;
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    if (image.pixelIndex(x + kx, y + ky) == 1) {
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }

            result.setPixel(x, y, found ? 1 : 0);
        }
    }
    return result;
}
