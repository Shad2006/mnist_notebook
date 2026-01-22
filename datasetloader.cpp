#include "datasetloader.h"
#include "charactermap.h"
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

    QStringList classDirs = datasetDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);

    int totalSamples = 0;
    QMap<QString, int> classCounts;

    for (const QString &classDirName : classDirs) {
        bool isDigit = false;
        int digit = -1;

        if (classDirName.length() == 1 && classDirName >= "0" && classDirName <= "9") {
            digit = classDirName.toInt();
            isDigit = true;
        }

        bool isCyrillic = false;
        QString cyrillicLetter;
        if (classDirName.length() == 1 && classDirName >= "А" && classDirName <= "Я") {
            cyrillicLetter = classDirName;
            isCyrillic = true;
        }

        if (!isDigit && !isCyrillic) {
            continue;
        }

        QDir classDir(datasetPath + "/" + classDirName);
        QStringList imageFiles = classDir.entryList({"*.png", "*.jpg", "*.jpeg", "*.bmp"}, QDir::Files);

        for (const QString &imageFile : imageFiles) {
            QString imagePath = classDir.absoluteFilePath(imageFile);
            QImage image(imagePath);

            if (image.isNull()) continue;

            QImage processed = preprocessImage(image);
            QVector<double> input = imageToVector(processed);

            QVector<double> target(43, 0.0);

            if (isDigit) {
                target[digit] = 1.0;
            } else if (isCyrillic) {
                int index = CharacterMap::charToIndex(cyrillicLetter);
                if (index >= 0) {
                    target[index] = 1.0;
                } else {
                    continue;
                }
            }

            TrainingData data;
            data.input = input;
            data.target = target;
            data.label = classDirName;

            m_trainingData.append(data);
            totalSamples++;
            classCounts[classDirName]++;
        }
    }

    qDebug() << "Загружено образцов:" << totalSamples;
    qDebug() << "Распределение по классам:";
    for (auto it = classCounts.begin(); it != classCounts.end(); ++it) {
        qDebug() << "  " << it.key() << ":" << it.value();
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
    QImage scaled = image.scaled(28, 28, Qt::KeepAspectRatio, Qt::SmoothTransformation);
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
            QVector<double> target(43, 0.0);
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
