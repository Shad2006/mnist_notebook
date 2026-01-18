#include "datasetloader.h"
#include <QDir>
#include <QFileInfo>
#include <QDebug>
#include <QPainter>
#include <QRegularExpression>
#include <QRegularExpressionMatch>
DatasetLoader::DatasetLoader() {
}

bool DatasetLoader::loadCustomDataset(const QString &datasetPath)
{
    m_trainingData.clear();

    QDir datasetDir(datasetPath);
    if (!datasetDir.exists()) {
        qDebug() << "Директория датасета не существует:" << datasetPath;
        return false;
    }
    QStringList imageFiles = datasetDir.entryList(
        {"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"},
        QDir::Files
        );
    int totalSamples = 0;
    int skippedFiles = 0;
    QRegularExpression regex("^(\\d+)\\\\(\\d+)\\.(png|jpg|jpeg|bmp|tiff)$");
    regex.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
    for (const QString &imageFile : imageFiles) {
        QRegularExpressionMatch match = regex.match(imageFile);
        if (!match.hasMatch()) {
            QRegularExpression altRegex1("^(\\d+)_(\\d+)\\.(png|jpg|jpeg|bmp|tiff)$");
            QRegularExpression altRegex2("^(\\d+)-(\\d+)\\.(png|jpg|jpeg|bmp|tiff)$");
            QRegularExpression altRegex3("^(\\d+)(\\d{4,})\\.(png|jpg|jpeg|bmp|tiff)$");
            if (altRegex1.match(imageFile).hasMatch()) {
                match = altRegex1.match(imageFile);
            } else if (altRegex2.match(imageFile).hasMatch()) {
                match = altRegex2.match(imageFile);
            } else if (altRegex3.match(imageFile).hasMatch()) {
                match = altRegex3.match(imageFile);
            } else {
                qDebug() << "Пропущен файл (не соответствует шаблону):" << imageFile;
                skippedFiles++;
                continue;
            }
        }
        QString digitStr = match.captured(1);
        bool ok;
        int digit = digitStr.toInt(&ok);
        if (!ok || digit < 0 || digit > 9) {
            qDebug() << "Пропущен файл (цифра не в диапазоне 0-9):" << imageFile;
            skippedFiles++;
            continue;
        }
        QString imagePath = datasetDir.absoluteFilePath(imageFile);
        QImage image(imagePath);
        if (image.isNull()) {
            qDebug() << "Не удалось загрузить изображение:" << imagePath;
            skippedFiles++;
            continue;
        }
        QImage processed = preprocessImage(image);
        QVector<double> input = imageToVector(processed);
        QVector<double> target(10, 0.0);
        target[digit] = 1.0;
        TrainingData data;
        data.input = input;
        data.target = target;
        data.label = QString::number(digit);
        m_trainingData.append(data);
        totalSamples++;
        if (totalSamples % 100 == 0) {
            qDebug() << "Загружено" << totalSamples << "образцов...";
        }
    }
    qDebug() << "Загрузка завершена!";
    qDebug() << "Успешно загружено:" << totalSamples << "образцов";
    qDebug() << "Пропущено файлов:" << skippedFiles;
    if (skippedFiles > 0) {
        qDebug() << "Поддерживаемые форматы имен файлов:";
        qDebug() << "1. 0\\001.png  (цифра\\номер.расширение)";
        qDebug() << "2. 0_001.png  (цифра_номер.расширение)";
        qDebug() << "3. 0-001.png  (цифра-номер.расширение)";
        qDebug() << "4. 00001.png  (цифраномер.расширение)";
    }
    return totalSamples > 0;
}
QVector<double> DatasetLoader::imageToVector(const QImage &image)
{
    QVector<double> result(784);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            QRgb pixel = image.pixel(x, y);
            int gray = qGray(pixel);
            double value = (255.0 - gray) / 255.0;
            result[y * 28 + x] = value;
        }
    }

    return result;
}

QImage DatasetLoader::preprocessImage(const QImage &image)
{
    QImage gray;
    if (image.format() != QImage::Format_Grayscale8) {
        gray = image.convertToFormat(QImage::Format_Grayscale8);
    } else {
        gray = image.copy();
    }

    int totalGray = 0;
    int pixelCount = gray.width() * gray.height();

    for (int y = 0; y < gray.height(); y++) {
        for (int x = 0; x < gray.width(); x++) {
            totalGray += qGray(gray.pixel(x, y));
        }
    }

    int threshold = totalGray / pixelCount;
    QImage binary(gray.size(), QImage::Format_Grayscale8);

    for (int y = 0; y < gray.height(); y++) {
        for (int x = 0; x < gray.width(); x++) {
            int grayValue = qGray(gray.pixel(x, y));
            int binaryValue = (grayValue < threshold) ? 0 : 255;
            binary.setPixel(x, y, qRgb(binaryValue, binaryValue, binaryValue));
        }
    }

    int minX = binary.width();
    int maxX = 0;
    int minY = binary.height();
    int maxY = 0;

    for (int y = 0; y < binary.height(); y++) {
        for (int x = 0; x < binary.width(); x++) {
            if (qGray(binary.pixel(x, y)) < 128) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    int padding = 2;
    int digitWidth = qMax(1, maxX - minX + 1);
    int digitHeight = qMax(1, maxY - minY + 1);

    QImage digitImage = binary.copy(
        qMax(0, minX - padding),
        qMax(0, minY - padding),
        qMin(binary.width(), digitWidth + 2 * padding),
        qMin(binary.height(), digitHeight + 2 * padding)
        );

    QImage scaled = digitImage.scaled(20, 20, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    QImage result(28, 28, QImage::Format_Grayscale8);
    result.fill(255);

    QPainter painter(&result);
    int xOffset = (28 - scaled.width()) / 2;
    int yOffset = (28 - scaled.height()) / 2;
    painter.drawImage(xOffset, yOffset, scaled);
    painter.end();

    return result;
}

bool DatasetLoader::loadOldDataset(const QString &datasetPath)
{
    m_trainingData.clear();

    QDir datasetDir(datasetPath);
    if (!datasetDir.exists()) {
        qDebug() << "Директория датасета не существует:" << datasetPath;
        return false;
    }

    QStringList digitDirs = datasetDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);

    int totalSamples = 0;

    for (const QString &digitDirName : digitDirs) {
        bool ok;
        int digit = digitDirName.toInt(&ok);

        if (!ok || digit < 0 || digit > 9) {
            continue;
        }

        QDir digitDir(datasetPath + "/" + digitDirName);
        QStringList imageFiles = digitDir.entryList({"*.png", "*.jpg", "*.bmp"}, QDir::Files);

        for (const QString &imageFile : imageFiles) {
            QString imagePath = digitDir.absoluteFilePath(imageFile);
            QImage image(imagePath);

            if (image.isNull()) {
                continue;
            }

            QImage processed = preprocessImage(image);
            QVector<double> input = imageToVector(processed);
            QVector<double> target(10, 0.0);
            target[digit] = 1.0;

            TrainingData data;
            data.input = input;
            data.target = target;
            data.label = QString::number(digit);

            m_trainingData.append(data);
            totalSamples++;
        }
    }
    qDebug() << "Загружено" << totalSamples << "образцов из старого датасета";
    return totalSamples > 0;
}
