#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QVector>
#include "commontypes.h"

class QLabel;
class QPushButton;
class QTextEdit;
class QProgressBar;
class QTableWidget;
class QListWidget;
class QSpinBox;
class QCheckBox;

class GridDetector;
class NeuralNetwork;
class DatasetLoader;
class CellProcessor;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void loadImage();
    void loadDataset();
    void trainModel();
    void recognizeCells();
    void saveResults();
    void clearAll();
    void applyCorrection(int row);
    void retrainOnCorrections();
    void updateManualGrid();
    void addManualGrid();
    void showCellDetails(int row, int column);
    void copyTextToClipboard();
    void clearTextEditor();
    void onTextEditorChanged();

private:
    void setupUI();
    void updateImageDisplay();
    void updateResultsTable();
    void logMessage(const QString &message);
    void performRecognition();
    void saveCorrectedSample(const QImage& cellImage, const QString& correctChar);
    void updateTextEditor();
    QImage cleanImageForGrid(const QImage &image, bool removeBlue, bool binarize, int threshold);

    QLabel *m_imageLabel;
    QLabel *m_processedImageLabel;
    QPushButton *m_loadImageButton;
    QPushButton *m_loadDatasetButton;
    QPushButton *m_trainButton;
    QPushButton *m_recognizeButton;
    QPushButton *m_saveButton;
    QPushButton *m_clearButton;
    QPushButton *m_retrainButton;
    QPushButton *m_addGridButton;
    QPushButton *m_updateGridButton;
    QPushButton *m_copyTextButton;
    QPushButton *m_clearTextButton;
    QProgressBar *m_progressBar;
    QTextEdit *m_logText;
    QTextEdit *m_textEditor;
    QTableWidget *m_resultsTable;
    QListWidget *m_cellList;
    QCheckBox *m_removeBlueLinesCheck;
    QCheckBox *m_binarizeCheck;
    QSpinBox *m_rowsSpinBox;
    QSpinBox *m_colsSpinBox;
    QSpinBox *m_marginSpinBox;
    QSpinBox *m_thresholdSpin;

    QImage m_originalImage;
    QImage m_processedImage;

    GridDetector *m_gridDetector;
    NeuralNetwork *m_neuralNetwork;
    DatasetLoader *m_datasetLoader;
    CellProcessor *m_cellProcessor;

    GridDetectionResult m_gridResult;
    QVector<RecognizedCell> m_recognizedCells;
    QVector<TrainingData> m_corrections;
};

#endif
