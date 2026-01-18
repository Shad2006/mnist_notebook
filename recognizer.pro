QT += core gui widgets
CONFIG += c++11
TARGET = recognizer
TEMPLATE = app

SOURCES = main.cpp \
    mainwindow.cpp \
    griddetector.cpp \
    datasetloader.cpp \
    neuralnetwork.cpp \
    cellprocessor.cpp

HEADERS = mainwindow.h \
    griddetector.h \
    datasetloader.h \
    neuralnetwork.h \
    cellprocessor.h \
    commontypes.h

RESOURCES +=
