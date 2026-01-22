#include "charactermap.h"
#include <QString>

QMap<QString, int> CharacterMap::s_charToIndex = CharacterMap::createCharToIndex();
QMap<int, QString> CharacterMap::s_indexToChar = CharacterMap::createIndexToChar();

QMap<QString, int> CharacterMap::createCharToIndex() {
    QMap<QString, int> map;

    for (int i = 0; i <= 9; i++) {
        map[QString::number(i)] = i;
    }

    QString cyrillic = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ";
    for (int i = 0; i < cyrillic.length(); i++) {
        map[cyrillic.mid(i, 1)] = 10 + i;
    }

    return map;
}

QMap<int, QString> CharacterMap::createIndexToChar() {
    QMap<int, QString> map;

    for (int i = 0; i <= 9; i++) {
        map[i] = QString::number(i);
    }

    QString cyrillic = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ";
    for (int i = 0; i < cyrillic.length(); i++) {
        map[10 + i] = cyrillic.mid(i, 1);
    }

    return map;
}

int CharacterMap::charToIndex(const QString &ch) {
    if (s_charToIndex.contains(ch)) {
        return s_charToIndex[ch];
    }
    return -1;
}

QString CharacterMap::indexToChar(int index) {
    if (s_indexToChar.contains(index)) {
        return s_indexToChar[index];
    }
    return "?";
}

bool CharacterMap::isDigit(const QString &ch) {
    return ch >= "0" && ch <= "9";
}

bool CharacterMap::isCyrillic(const QString &ch) {
    return (ch >= "А" && ch <= "Я");
}
