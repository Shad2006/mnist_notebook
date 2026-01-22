#ifndef CHARACTERMAP_H
#define CHARACTERMAP_H

#include <QString>
#include <QMap>

class CharacterMap {
public:
    CharacterMap();

    static int charToIndex(const QString &ch);
    static QString indexToChar(int index);
    static int numClasses() { return 43; }

    static bool isDigit(const QString &ch);
    static bool isCyrillic(const QString &ch);

private:
    static QMap<QString, int> createCharToIndex();
    static QMap<int, QString> createIndexToChar();

    static QMap<QString, int> s_charToIndex;
    static QMap<int, QString> s_indexToChar;
};

#endif
