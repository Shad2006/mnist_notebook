#include "mainwindow.h"
#include "griddetector.h"
#include "neuralnetwork.h"
#include "datasetloader.h"
#include "cellprocessor.h"
#include "charactermap.h"
#include "commontypes.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QProgressBar>
#include <QTableWidget>
#include <QListWidget>
#include <QHeaderView>
#include <QFileDialog>
#include <QMessageBox>
#include <QImageReader>
#include <QDateTime>
#include <QDir>
#include <QLineEdit>
#include <QSpinBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QGroupBox>
#include <QPainter>
#include <QBrush>
#include <QFile>
#include <QTextStream>
#include <QCheckBox>
#include <QClipboard>
#include <QApplication>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_gridDetector(new GridDetector())
    , m_neuralNetwork(new NeuralNetwork(784, 256, 128, 43))
    , m_datasetLoader(new DatasetLoader())
    , m_cellProcessor(new CellProcessor())
{
    setupUI();

    if (m_neuralNetwork->loadModel("model_improved.dat")) {
        logMessage("Модель загружена из model_improved.dat");
        logMessage(m_neuralNetwork->getModelInfo());
    } else {
        logMessage("Модель не найдена. Необходимо обучение.");
    }

    QFile correctionsFile("corrections.dat");
    if (correctionsFile.exists()) {
        m_neuralNetwork->loadCorrections("corrections.dat", m_corrections);
        logMessage(QString("Загружено %1 исправлений").arg(m_corrections.size()));
    }

    setWindowTitle("43Characters");
    resize(1400, 900);
}

MainWindow::~MainWindow()
{
    m_neuralNetwork->saveModel("model_improved.dat");
    m_neuralNetwork->saveCorrections("corrections.dat", m_corrections);

    delete m_gridDetector;
    delete m_neuralNetwork;
    delete m_datasetLoader;
    delete m_cellProcessor;
}

void MainWindow::setupUI()
{
    QWidget *centralWidget = new QWidget;
    setCentralWidget(centralWidget);

    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    QHBoxLayout *buttonLayout = new QHBoxLayout;

    m_loadImageButton = new QPushButton("Загрузить изображение");
    m_loadDatasetButton = new QPushButton("Загрузить датасет");
    m_trainButton = new QPushButton("Обучить модель");
    m_recognizeButton = new QPushButton("Распознать клетки");
    m_saveButton = new QPushButton("Сохранить результаты");
    m_clearButton = new QPushButton("Очистить всё");
    m_retrainButton = new QPushButton("Переобучить на исправлениях");
    m_retrainButton->setEnabled(false);
    m_retrainButton->setStyleSheet("background-color: #FFE4B5;");

    buttonLayout->addWidget(m_loadImageButton);
    buttonLayout->addWidget(m_loadDatasetButton);
    buttonLayout->addWidget(m_trainButton);
    buttonLayout->addWidget(m_recognizeButton);
    buttonLayout->addWidget(m_saveButton);
    buttonLayout->addWidget(m_clearButton);
    buttonLayout->addWidget(m_retrainButton);

    mainLayout->addLayout(buttonLayout);

    QHBoxLayout *manualGridLayout = new QHBoxLayout;
    manualGridLayout->addWidget(new QLabel("Ручная сетка:"));

    manualGridLayout->addWidget(new QLabel("Строки:"));
    m_rowsSpinBox = new QSpinBox;
    m_rowsSpinBox->setRange(1, 50);
    m_rowsSpinBox->setValue(5);
    manualGridLayout->addWidget(m_rowsSpinBox);

    manualGridLayout->addWidget(new QLabel("Колонки:"));
    m_colsSpinBox = new QSpinBox;
    m_colsSpinBox->setRange(1, 50);
    m_colsSpinBox->setValue(5);
    manualGridLayout->addWidget(m_colsSpinBox);

    manualGridLayout->addWidget(new QLabel("Отступ:"));
    m_marginSpinBox = new QSpinBox;
    m_marginSpinBox->setRange(0, 50);
    m_marginSpinBox->setValue(10);
    manualGridLayout->addWidget(m_marginSpinBox);

    m_removeBlueLinesCheck = new QCheckBox("Удалить синие линии");
    m_removeBlueLinesCheck->setChecked(true);
    manualGridLayout->addWidget(m_removeBlueLinesCheck);

    m_binarizeCheck = new QCheckBox("Бинаризовать");
    m_binarizeCheck->setChecked(true);
    manualGridLayout->addWidget(m_binarizeCheck);

    manualGridLayout->addWidget(new QLabel("Порог:"));
    m_thresholdSpin = new QSpinBox;
    m_thresholdSpin->setRange(0, 255);
    m_thresholdSpin->setValue(100);
    m_thresholdSpin->setMaximumWidth(60);
    manualGridLayout->addWidget(m_thresholdSpin);

    m_addGridButton = new QPushButton("+ Добавить сетку");
    m_addGridButton->setStyleSheet("background-color: #90EE90;");
    manualGridLayout->addWidget(m_addGridButton);

    m_updateGridButton = new QPushButton("Обновить распознавание");
    m_updateGridButton->setStyleSheet("background-color: #ADD8E6;");
    manualGridLayout->addWidget(m_updateGridButton);

    manualGridLayout->addStretch();
    mainLayout->addLayout(manualGridLayout);

    m_progressBar = new QProgressBar;
    m_progressBar->setVisible(false);
    mainLayout->addWidget(m_progressBar);

    QHBoxLayout *imageLayout = new QHBoxLayout;

    QVBoxLayout *originalLayout = new QVBoxLayout;
    originalLayout->addWidget(new QLabel("Исходное изображение:"));
    m_imageLabel = new QLabel;
    m_imageLabel->setMinimumSize(400, 300);
    m_imageLabel->setAlignment(Qt::AlignCenter);
    m_imageLabel->setStyleSheet("border: 1px solid gray;");
    originalLayout->addWidget(m_imageLabel);

    QVBoxLayout *processedLayout = new QVBoxLayout;
    processedLayout->addWidget(new QLabel("Обработанное изображение:"));
    m_processedImageLabel = new QLabel;
    m_processedImageLabel->setMinimumSize(400, 300);
    m_processedImageLabel->setAlignment(Qt::AlignCenter);
    m_processedImageLabel->setStyleSheet("border: 1px solid gray;");
    processedLayout->addWidget(m_processedImageLabel);

    imageLayout->addLayout(originalLayout);
    imageLayout->addLayout(processedLayout);

    mainLayout->addLayout(imageLayout);

    QHBoxLayout *resultsLayout = new QHBoxLayout;

    m_resultsTable = new QTableWidget;
    m_resultsTable->setColumnCount(7);
    m_resultsTable->setHorizontalHeaderLabels({
        "Клетка",
        "Символ",
        "Уверенность",
        "Изображение",
        "Исправить на",
        "Действие",
        "Статус"
    });
    m_resultsTable->horizontalHeader()->setStretchLastSection(false);
    m_resultsTable->setColumnWidth(0, 80);
    m_resultsTable->setColumnWidth(1, 60);
    m_resultsTable->setColumnWidth(2, 100);
    m_resultsTable->setColumnWidth(3, 80);
    m_resultsTable->setColumnWidth(4, 100);
    m_resultsTable->setColumnWidth(5, 100);
    m_resultsTable->setColumnWidth(6, 100);
    m_resultsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    connect(m_resultsTable, &QTableWidget::cellDoubleClicked,
            this, &MainWindow::showCellDetails);

    resultsLayout->addWidget(m_resultsTable, 3);

    QVBoxLayout *sideLayout = new QVBoxLayout;
    sideLayout->addWidget(new QLabel("Лог выполнения:"));
    m_logText = new QTextEdit;
    m_logText->setMaximumHeight(200);
    m_logText->setReadOnly(true);
    sideLayout->addWidget(m_logText);

    sideLayout->addWidget(new QLabel("Обнаруженные клетки:"));
    m_cellList = new QListWidget;
    sideLayout->addWidget(m_cellList);

    resultsLayout->addLayout(sideLayout, 1);
    mainLayout->addLayout(resultsLayout);

    QVBoxLayout *textEditorLayout = new QVBoxLayout;
    textEditorLayout->addWidget(new QLabel("Распознанный текст:"));

    m_textEditor = new QTextEdit;
    m_textEditor->setMaximumHeight(150);
    m_textEditor->setReadOnly(false);
    textEditorLayout->addWidget(m_textEditor);

    QHBoxLayout *textButtonsLayout = new QHBoxLayout;
    m_copyTextButton = new QPushButton("Копировать текст");
    m_clearTextButton = new QPushButton("Очистить текст");

    textButtonsLayout->addWidget(m_copyTextButton);
    textButtonsLayout->addWidget(m_clearTextButton);
    textButtonsLayout->addStretch();

    textEditorLayout->addLayout(textButtonsLayout);
    mainLayout->addLayout(textEditorLayout);

    connect(m_loadImageButton, &QPushButton::clicked, this, &MainWindow::loadImage);
    connect(m_loadDatasetButton, &QPushButton::clicked, this, &MainWindow::loadDataset);
    connect(m_trainButton, &QPushButton::clicked, this, &MainWindow::trainModel);
    connect(m_recognizeButton, &QPushButton::clicked, this, &MainWindow::recognizeCells);
    connect(m_saveButton, &QPushButton::clicked, this, &MainWindow::saveResults);
    connect(m_clearButton, &QPushButton::clicked, this, &MainWindow::clearAll);
    connect(m_retrainButton, &QPushButton::clicked, this, &MainWindow::retrainOnCorrections);
    connect(m_addGridButton, &QPushButton::clicked, this, &MainWindow::addManualGrid);
    connect(m_updateGridButton, &QPushButton::clicked, this, &MainWindow::updateManualGrid);
    connect(m_copyTextButton, &QPushButton::clicked, this, &MainWindow::copyTextToClipboard);
    connect(m_clearTextButton, &QPushButton::clicked, this, &MainWindow::clearTextEditor);
    connect(m_textEditor, &QTextEdit::textChanged, this, &MainWindow::onTextEditorChanged);
}

void MainWindow::loadImage()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Выберите изображение", "",
                                                    "Images (*.png *.jpg *.jpeg *.bmp *.tiff)");

    if (fileName.isEmpty()) {
        return;
    }

    QImage image(fileName);
    if (image.isNull()) {
        QMessageBox::warning(this, "Ошибка", "Не удалось загрузить изображение");
        return;
    }

    m_originalImage = image;
    updateImageDisplay();

    logMessage(QString("Изображение загружено: %1").arg(fileName));
}

void MainWindow::loadDataset()
{
    QString datasetPath = QFileDialog::getExistingDirectory(this, "Выберите папку с датасетом");

    if (datasetPath.isEmpty()) {
        return;
    }

    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0);

    bool success = m_datasetLoader->loadCustomDataset(datasetPath);

    m_progressBar->setVisible(false);

    if (success) {
        int sampleCount = m_datasetLoader->getSampleCount();
        logMessage(QString("Датасет загружен: %1 образцов").arg(sampleCount));
    } else {
        QMessageBox::warning(this, "Ошибка", "Не удалось загрузить датасет");
    }
}

void MainWindow::trainModel()
{
    QVector<TrainingData> dataset = m_datasetLoader->getTrainingData();

    if (dataset.isEmpty()) {
        QMessageBox::information(this, "Информация", "Сначала загрузите датасет");
        return;
    }

    dataset.append(m_corrections);

    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 15);

    for (int epoch = 0; epoch < 15; epoch++) {
        m_neuralNetwork->trainOnDataset(dataset, 1);
        m_progressBar->setValue(epoch + 1);
        QCoreApplication::processEvents();
    }

    m_progressBar->setVisible(false);

    if (m_neuralNetwork->saveModel("model_improved.dat")) {
        logMessage("Улучшенная модель обучена и сохранена в model_improved.dat");
        logMessage(m_neuralNetwork->getModelInfo());
    }
}

void MainWindow::recognizeCells()
{
    if (m_originalImage.isNull()) {
        QMessageBox::information(this, "Информация", "Сначала загрузите изображение");
        return;
    }

    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0);

    logMessage("Детектирование сетки...");
    m_gridResult = m_gridDetector->detectGrid(m_originalImage);

    if (m_gridResult.cells.isEmpty()) {
        m_progressBar->setVisible(false);
        QMessageBox::warning(this, "Ошибка", "Не удалось обнаружить сетку");
        return;
    }

    m_processedImage = m_gridResult.processedImage;
    updateImageDisplay();

    performRecognition();
}

void MainWindow::performRecognition()
{
    m_recognizedCells.clear();
    m_cellList->clear();

    logMessage(QString("Найдено %1 клеток. Распознавание...").arg(m_gridResult.cells.size()));

    for (int i = 0; i < m_gridResult.cells.size(); i++) {
        QImage cellImage = m_gridResult.cellImages[i];

        QVector<double> input = m_cellProcessor->preprocessCell(cellImage);
        QImage processedImage = m_cellProcessor->getProcessedImage(cellImage);

        int predicted = m_neuralNetwork->predict(input);
        QVector<double> outputs = m_neuralNetwork->getOutputs();
        double confidence = outputs[predicted];

        QString character = CharacterMap::indexToChar(predicted);

        RecognizedCell cell;
        cell.character = character;
        cell.confidence = confidence;
        cell.boundingRect = m_gridResult.cells[i];
        cell.cellImage = cellImage;
        cell.processedImage = processedImage;
        cell.inputVector = input;
        cell.corrected = false;
        cell.row = 0;
        cell.col = 0;

        m_recognizedCells.append(cell);

        m_cellList->addItem(QString("Клетка %1: %2 (%3%)")
                                .arg(i + 1)
                                .arg(cell.character)
                                .arg(qRound(cell.confidence * 100)));
    }

    if (!m_recognizedCells.isEmpty()) {
        QVector<QPoint> cellPositions;
        for (int i = 0; i < m_gridResult.cells.size(); i++) {
            cellPositions.append(m_gridResult.cells[i].topLeft());
        }

        QSet<int> uniqueYs;
        for (const QPoint &pos : cellPositions) {
            uniqueYs.insert(pos.y());
        }
        QList<int> sortedYs = uniqueYs.values();
        std::sort(sortedYs.begin(), sortedYs.end());

        QSet<int> uniqueXs;
        for (const QPoint &pos : cellPositions) {
            uniqueXs.insert(pos.x());
        }
        QList<int> sortedXs = uniqueXs.values();
        std::sort(sortedXs.begin(), sortedXs.end());

        for (int i = 0; i < m_recognizedCells.size(); i++) {
            QPoint pos = m_gridResult.cells[i].topLeft();

            int row = 0;
            for (int y : sortedYs) {
                if (qAbs(pos.y() - y) < 10) {
                    break;
                }
                row++;
            }

            int col = 0;
            for (int x : sortedXs) {
                if (qAbs(pos.x() - x) < 10) {
                    break;
                }
                col++;
            }

            m_recognizedCells[i].row = row;
            m_recognizedCells[i].col = col;
        }
    }

    m_progressBar->setVisible(false);
    updateResultsTable();
    updateTextEditor();

    logMessage(QString("Распознавание завершено. Распознано %1 клеток").arg(m_recognizedCells.size()));
}

void MainWindow::updateResultsTable()
{
    m_resultsTable->clearContents();
    m_resultsTable->setRowCount(m_recognizedCells.size());

    for (int i = 0; i < m_recognizedCells.size(); i++) {
        const RecognizedCell &cell = m_recognizedCells[i];

        QTableWidgetItem *cellItem = new QTableWidgetItem(QString("Клетка %1").arg(i + 1));

        QTableWidgetItem *charItem = new QTableWidgetItem(cell.character);
        charItem->setTextAlignment(Qt::AlignCenter);
        if (cell.confidence < 0.7) {
            charItem->setBackground(QBrush(QColor(255, 200, 200)));
        }

        QTableWidgetItem *confidenceItem = new QTableWidgetItem(
            QString("%1%").arg(qRound(cell.confidence * 100)));
        confidenceItem->setTextAlignment(Qt::AlignCenter);

        QLabel *imageLabel = new QLabel;
        QPixmap pixmap = QPixmap::fromImage(cell.processedImage.scaled(56, 56));
        imageLabel->setPixmap(pixmap);
        imageLabel->setAlignment(Qt::AlignCenter);
        imageLabel->setToolTip("Двойной клик для увеличения");

        QLineEdit *correctionEdit = new QLineEdit;
        correctionEdit->setPlaceholderText("Символ");
        correctionEdit->setMaxLength(1);
        correctionEdit->setAlignment(Qt::AlignCenter);
        correctionEdit->setMaximumWidth(50);

        QPushButton *correctButton = new QPushButton("Исправить");
        correctButton->setProperty("row", i);
        correctButton->setMaximumWidth(80);
        connect(correctButton, &QPushButton::clicked, [this, i]() {
            applyCorrection(i);
        });

        QTableWidgetItem *statusItem = new QTableWidgetItem(cell.corrected ? "Исправлена" : "");
        if (cell.corrected) {
            statusItem->setBackground(QBrush(QColor(200, 255, 200)));
            correctionEdit->setText(cell.character);
        }

        m_resultsTable->setItem(i, 0, cellItem);
        m_resultsTable->setItem(i, 1, charItem);
        m_resultsTable->setItem(i, 2, confidenceItem);
        m_resultsTable->setCellWidget(i, 3, imageLabel);
        m_resultsTable->setCellWidget(i, 4, correctionEdit);
        m_resultsTable->setCellWidget(i, 5, correctButton);
        m_resultsTable->setItem(i, 6, statusItem);
    }

    bool hasCorrections = false;
    for (const auto& cell : m_recognizedCells) {
        if (cell.corrected) {
            hasCorrections = true;
            break;
        }
    }
    m_retrainButton->setEnabled(hasCorrections || !m_corrections.isEmpty());
}

void MainWindow::applyCorrection(int row)
{
    if (row < 0 || row >= m_recognizedCells.size()) return;

    QLineEdit *edit = qobject_cast<QLineEdit*>(m_resultsTable->cellWidget(row, 4));
    if (!edit) return;

    QString correction = edit->text().trimmed();

    if (correction.isEmpty()) {
        QMessageBox::information(this, "Информация", "Введите символ");
        return;
    }

    if (correction.length() != 1) {
        QMessageBox::warning(this, "Ошибка", "Введите один символ");
        return;
    }

    QChar ch = correction[0];
    bool isValid = false;

    if (ch.isDigit()) {
        int digit = ch.unicode() - '0';
        if (digit >= 0 && digit <= 9) {
            isValid = true;
        }
    } else if (ch >= QChar(0x0410) && ch <= QChar(0x042F)) {
        isValid = true;
    }

    if (!isValid) {
        QMessageBox::warning(this, "Ошибка", "Введите цифру (0-9) или букву (А-Я)");
        return;
    }

    m_recognizedCells[row].character = correction;
    m_recognizedCells[row].corrected = true;

    saveCorrectedSample(m_recognizedCells[row].cellImage, correction);

    m_resultsTable->item(row, 1)->setText(correction);
    m_resultsTable->item(row, 1)->setBackground(QBrush(QColor(200, 255, 200)));
    m_resultsTable->item(row, 6)->setText("Исправлена");
    m_resultsTable->item(row, 6)->setBackground(QBrush(QColor(200, 255, 200)));

    m_cellList->item(row)->setText(QString("Клетка %1: %2 (ИСПРАВЛЕНО)")
                                       .arg(row + 1)
                                       .arg(correction));

    m_retrainButton->setEnabled(true);
    updateTextEditor();

    logMessage(QString("Клетка %1 исправлена: %2")
                   .arg(row + 1)
                   .arg(correction));
}

void MainWindow::saveCorrectedSample(const QImage& cellImage, const QString& correctChar)
{
    QImage processed = m_cellProcessor->getProcessedImage(cellImage);
    QVector<double> input = m_cellProcessor->preprocessCell(cellImage);

    int index = CharacterMap::charToIndex(correctChar);
    if (index == -1) {
        qDebug() << "Неизвестный символ для сохранения:" << correctChar;
        return;
    }

    QVector<double> target(43, 0.0);
    target[index] = 1.0;

    TrainingData data;
    data.input = input;
    data.target = target;
    data.label = correctChar;

    m_corrections.append(data);

    logMessage(QString("Добавлено исправление для символа %1").arg(correctChar));
}

void MainWindow::retrainOnCorrections()
{
    if (m_corrections.isEmpty()) {
        QMessageBox::information(this, "Информация", "Нет исправлений для переобучения");
        return;
    }

    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 5);

    logMessage(QString("Переобучение на %1 исправленных образцах...").arg(m_corrections.size()));

    for (int epoch = 0; epoch < 5; epoch++) {
        m_neuralNetwork->trainOnDataset(m_corrections, 1);
        m_progressBar->setValue(epoch + 1);
        QCoreApplication::processEvents();
    }

    m_progressBar->setVisible(false);

    if (m_neuralNetwork->saveCorrections("corrections.dat", m_corrections)) {
        logMessage("Исправления сохранены в corrections.dat");
    }

    performRecognition();
    logMessage("Переобучение завершено");
}

void MainWindow::addManualGrid()
{
    if (m_originalImage.isNull()) {
        QMessageBox::information(this, "Информация", "Сначала загрузите изображение");
        return;
    }

    int rows = m_rowsSpinBox->value();
    int cols = m_colsSpinBox->value();
    int margin = m_marginSpinBox->value();

    bool removeBlue = m_removeBlueLinesCheck->isChecked();
    bool binarize = m_binarizeCheck->isChecked();
    int threshold = m_thresholdSpin->value();

    m_gridResult = GridDetectionResult();

    QImage cleanedImage = m_originalImage.copy();

    if (removeBlue || binarize) {
        cleanedImage = cleanImageForGrid(cleanedImage, removeBlue, binarize, threshold);
    }

    QImage processedImage = cleanedImage.copy();
    QPainter painter(&processedImage);
    painter.setPen(QPen(QColor(0, 100, 255), 2));

    int cellWidth = cleanedImage.width() / cols;
    int cellHeight = cleanedImage.height() / rows;

    for (int i = 0; i <= rows; i++) {
        int y = i * cellHeight;
        painter.drawLine(0, y, processedImage.width(), y);
    }

    for (int i = 0; i <= cols; i++) {
        int x = i * cellWidth;
        painter.drawLine(x, 0, x, processedImage.height());
    }

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int left = col * cellWidth + margin;
            int top = row * cellHeight + margin;
            int right = (col + 1) * cellWidth - margin;
            int bottom = (row + 1) * cellHeight - margin;

            if (right > left && bottom > top) {
                QRect cellRect(left, top, right - left, bottom - top);

                QImage cellImage = cleanedImage.copy(cellRect);

                if (!m_gridDetector->isCellEmpty(cellImage)) {
                    m_gridResult.cells.append(cellRect);
                    m_gridResult.cellImages.append(cellImage);

                    painter.setPen(QPen(Qt::green, 1));
                    painter.drawRect(cellRect);
                }
            }
        }
    }

    painter.end();
    m_gridResult.processedImage = processedImage;
    m_processedImage = processedImage;
    updateImageDisplay();

    logMessage(QString("Ручная сетка %1x%2 создана. Найдено %3 непустых клеток")
                   .arg(rows).arg(cols).arg(m_gridResult.cells.size()));

    performRecognition();
}

QImage MainWindow::cleanImageForGrid(const QImage &image, bool removeBlue, bool binarize, int threshold)
{
    QImage result = image.copy();

    for (int y = 0; y < image.height(); y++) {
        for (int x = 0; x < image.width(); x++) {
            QRgb pixel = image.pixel(x, y);
            int gray = qGray(pixel);

            if (removeBlue) {
                int r = qRed(pixel);
                int g = qGreen(pixel);
                int b = qBlue(pixel);

                bool isBlue = (b > r + 30 && b > g + 20) ||
                              (b > 150 && g > 120 && r < 100);

                if (isBlue) {
                    result.setPixel(x, y, qRgb(255, 255, 255));
                    continue;
                }
            }

            if (binarize) {
                if (gray < threshold) {
                    result.setPixel(x, y, qRgb(0, 0, 0));
                } else {
                    result.setPixel(x, y, qRgb(255, 255, 255));
                }
            }
        }
    }

    return result;
}

void MainWindow::updateManualGrid()
{
    if (m_originalImage.isNull()) {
        QMessageBox::information(this, "Информация", "Сначала загрузите изображение");
        return;
    }

    if (m_gridResult.cells.isEmpty()) {
        QMessageBox::information(this, "Информация", "Сначала создайте сетку");
        return;
    }

    performRecognition();
}

void MainWindow::showCellDetails(int row, int column)
{
    if (row < 0 || row >= m_recognizedCells.size()) return;

    if (column == 3) {
        QDialog dialog(this);
        dialog.setWindowTitle(QString("Клетка %1").arg(row + 1));

        QVBoxLayout *layout = new QVBoxLayout(&dialog);

        QLabel *imageLabel = new QLabel;
        QPixmap pixmap = QPixmap::fromImage(m_recognizedCells[row].processedImage.scaled(280, 280));
        imageLabel->setPixmap(pixmap);
        imageLabel->setAlignment(Qt::AlignCenter);

        layout->addWidget(imageLabel);

        QLabel *infoLabel = new QLabel(
            QString("Распознано: %1\nУверенность: %2%\nРазмер: %3x%4")
                .arg(m_recognizedCells[row].character)
                .arg(qRound(m_recognizedCells[row].confidence * 100))
                .arg(m_recognizedCells[row].boundingRect.width())
                .arg(m_recognizedCells[row].boundingRect.height())
            );
        layout->addWidget(infoLabel);

        dialog.exec();
    }
}

void MainWindow::saveResults()
{
    if (m_recognizedCells.isEmpty()) {
        QMessageBox::information(this, "Информация", "Нет результатов для сохранения");
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(this, "Сохранить результаты",
                                                    QString("results_%1.txt").arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss")),
                                                    "Text files (*.txt)");

    if (fileName.isEmpty()) {
        return;
    }

    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "Ошибка", "Не удалось открыть файл для записи");
        return;
    }

    QTextStream out(&file);
    out << "Результаты распознавания\n";
    out << "=======================\n\n";

    for (int i = 0; i < m_recognizedCells.size(); i++) {
        const RecognizedCell &cell = m_recognizedCells[i];
        out << QString("Клетка %1: %2 (уверенность: %3%)\n")
                   .arg(i + 1)
                   .arg(cell.character)
                   .arg(qRound(cell.confidence * 100));
    }

    out << "\n\nТекст:\n";
    out << "======\n\n";
    out << m_textEditor->toPlainText();

    file.close();

    logMessage(QString("Результаты сохранены в %1").arg(fileName));
}

void MainWindow::clearAll()
{
    m_originalImage = QImage();
    m_processedImage = QImage();
    m_gridResult = GridDetectionResult();
    m_recognizedCells.clear();

    m_imageLabel->clear();
    m_processedImageLabel->clear();
    m_resultsTable->clearContents();
    m_resultsTable->setRowCount(0);
    m_cellList->clear();
    m_logText->clear();
    m_textEditor->clear();

    logMessage("Все данные очищены");
}

void MainWindow::updateImageDisplay()
{
    if (!m_originalImage.isNull()) {
        QImage displayImage = m_originalImage.scaled(400, 300, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        m_imageLabel->setPixmap(QPixmap::fromImage(displayImage));
    }

    if (!m_processedImage.isNull()) {
        QImage displayImage = m_processedImage.scaled(400, 300, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        m_processedImageLabel->setPixmap(QPixmap::fromImage(displayImage));
    }
}

void MainWindow::logMessage(const QString &message)
{
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    m_logText->append(QString("[%1] %2").arg(timestamp, message));
}

void MainWindow::updateTextEditor()
{
    if (m_recognizedCells.isEmpty()) return;

    int maxRow = 0;
    int maxCol = 0;

    for (const RecognizedCell &cell : m_recognizedCells) {
        if (cell.row > maxRow) maxRow = cell.row;
        if (cell.col > maxCol) maxCol = cell.col;
    }

    QVector<QVector<QString>> grid(maxRow + 1, QVector<QString>(maxCol + 1, " "));

    for (const RecognizedCell &cell : m_recognizedCells) {
        if (cell.row <= maxRow && cell.col <= maxCol) {
            grid[cell.row][cell.col] = cell.character;
        }
    }

    QString text;
    for (int row = 0; row <= maxRow; row++) {
        for (int col = 0; col <= maxCol; col++) {
            text += grid[row][col];
            if (col < maxCol) text += " ";
        }
        text += "\n";
    }

    m_textEditor->setPlainText(text);
}

void MainWindow::copyTextToClipboard()
{
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(m_textEditor->toPlainText());
    logMessage("Текст скопирован в буфер обмена");
}

void MainWindow::clearTextEditor()
{
    m_textEditor->clear();
}

void MainWindow::onTextEditorChanged()
{
}
