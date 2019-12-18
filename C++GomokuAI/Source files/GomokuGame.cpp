#include "pch.h"
#include "GomokuGame.h"
#include "Utils.h"

#include <string.h>
#include <thread>
#include <future>
#include <iostream>

#define _EMPTYSYMBOL_ 0
#define _P1SYMBOL_ 1
#define _P2SYMBOL_ 2

/*--------------------------------------------------------------*/

GomokuGame::GomokuGame(short sideLength, short winAmount)
	: m_sideLength(sideLength)
	, m_winAmount(winAmount)
	, m_boardLength(m_sideLength * m_sideLength)
	, m_movesPlayed(0)
	, m_winner(WinnerState_enum::None)
	, m_playerTurn(true)
{
	m_gameBoard = new char[m_boardLength];

	for (int i = 0; i < m_boardLength; i++)
	{
		m_gameBoard[i] = _EMPTYSYMBOL_;
	}
}

GomokuGame::GomokuGame(GomokuGame const& other)
{
	m_sideLength = other.m_sideLength;
	m_winAmount = other.m_winAmount;
	m_boardLength = m_sideLength * m_sideLength;
	m_movesPlayed = other.m_movesPlayed;
	m_winner = other.m_winner;
	m_playerTurn = other.m_playerTurn;
	memcpy(m_gameBoard, other.m_gameBoard, m_boardLength * sizeof(char));
}

GomokuGame::~GomokuGame()
{
	delete m_gameBoard;
}

void GomokuGame::operator=(GomokuGame const& other)
{
	m_sideLength = other.m_sideLength;
	m_winAmount = other.m_winAmount;
	m_boardLength = m_sideLength * m_sideLength;
	m_movesPlayed = other.m_movesPlayed;
	m_winner = other.m_winner;
	m_playerTurn = other.m_playerTurn;
	memcpy(m_gameBoard, other.m_gameBoard, m_boardLength * sizeof(char));
}

/*--------------------------------------------------------------*/

void GomokuGame::ResetBoard()
{
	for (int i = 0; i < m_boardLength; i++)
	{
		m_gameBoard[i] = _EMPTYSYMBOL_;
	}
}

bool GomokuGame::IsBoardFull() const
{
	return m_movesPlayed == m_boardLength;
}

bool GomokuGame::IsMoveWinning(int index)
{
	return IsMoveWinning_(index);
}

bool GomokuGame::IsMoveWinning(short row, short col)
{
	return IsMoveWinning_(Utils::ConvertToIndex(row, col, m_sideLength));
}

char** GomokuGame::GetMatrix()
{
	char** matrix = new char*[m_sideLength];
	for (int i = 0; i < m_sideLength; i++)
	{
		matrix[i] = new char[m_sideLength];
	}

	for (int i = 0; i < m_boardLength; i++)
	{
		int row = i / m_sideLength;
		int col = i % m_sideLength;

		matrix[row][col] = m_gameBoard[i];
	}

	return matrix;
}

bool GomokuGame::PlayMove(short row, short col)
{
	int index = Utils::ConvertToIndex(row, col, m_sideLength);
	if (m_gameBoard[index] != _EMPTYSYMBOL_)
		return false;

	m_gameBoard[index] = m_playerTurn ? _P1SYMBOL_ : _P2SYMBOL_;
	m_movesPlayed++;

	if (IsMoveWinning_(index))
		m_winner = m_playerTurn ? WinnerState_enum::P1 : WinnerState_enum::P2;
	else if (IsBoardFull())
		m_winner = WinnerState_enum::Draw;

	m_playerTurn = !m_playerTurn;
	return true;
}

WinnerState_enum GomokuGame::GetGameWinState() const
{
	return m_winner;
}

/*--------------------------------------------------------------*/

void GomokuGame::FreeCurrentBoard_()
{
	delete m_gameBoard;
}

int GomokuGame::DirectionAmount_(
	int index,
	char symbol,
	std::function<int(int)> indexModifier,
	std::function<bool(int)> indexCheck) const
{
	int aboveAmount = 0;
	int testIndex = indexModifier(index);
	while (indexCheck)
	{
		if (m_gameBoard[testIndex] == symbol)
		{
			aboveAmount++;
			testIndex = indexModifier(testIndex);
		}
		else
			break;

		if (aboveAmount >= (m_winAmount - 1))
			break;
	}

	return aboveAmount;
}

bool GomokuGame::IsMoveWinning_(int index) const
{
	char symbol = m_gameBoard[index];
	if (symbol == _EMPTYSYMBOL_)
		return false;

	int sideLength = m_sideLength;
	int boardLength = m_boardLength;

	auto aboveModifier = [sideLength](int index) {return index - sideLength;};
	auto aboveCheck = [](int index) {return index >= 0; };
	std::future<int> abovePromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, aboveModifier, aboveCheck);

	auto belowModifier = [sideLength](int index) {return index + sideLength;};
	auto belowCheck = [boardLength](int index) {return index < boardLength;};
	std::future<int> belowPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, belowModifier, belowCheck);

	auto leftModifier = [](int index) {return index - 1; };
	auto leftCheck = [sideLength](int index) {return (index % sideLength) < (sideLength - 1);};
	std::future<int> leftPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, leftModifier, leftCheck);

	auto rightModifier = [](int index) {return index + 1; };
	auto rightCheck = [sideLength](int index) {return (index % sideLength) != 0;};
	std::future<int> rightPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, rightModifier, rightCheck);

	auto upLeftModifier = [sideLength](int index) {return index - sideLength - 1;};
	auto upLeftCheck = [sideLength](int index) {return index > 0 && (index % sideLength) < (sideLength - 1);};
	std::future<int> upLeftPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, upLeftModifier, upLeftCheck);

	auto upRightModifier = [sideLength](int index) {return index - sideLength + 1; };
	auto upRightCheck = [sideLength](int index) {return index > 0 && (index % sideLength) != 0;};
	std::future<int> upRightPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, upRightModifier, upRightCheck);

	auto downLeftModifier = [sideLength](int index) {return index + sideLength - 1; };
	auto downLeftCheck = [boardLength, sideLength](int index) {return index < boardLength && (index % sideLength) < (sideLength - 1);};
	std::future<int> downLeftPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, downLeftModifier, downLeftCheck);

	auto downRightModifier = [sideLength](int index) {return index + sideLength + 1; };
	auto downRightCheck = [boardLength, sideLength](int index) {return index < boardLength && (index % sideLength) != 0;};
	std::future<int> downRightPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, downRightModifier, downRightCheck);

	int above = abovePromise.get();
	int below = belowPromise.get();
	int left = leftPromise.get();
	int right = rightPromise.get();

	int upLeft = upLeftPromise.get();
	int upRight = upRightPromise.get();
	int downLeft = downLeftPromise.get();
	int downRight = downRightPromise.get();

	if (above + below + 1 >= m_winAmount)
		return true;
	else if (left + right + 1 >= m_winAmount)
		return true;
	else if (upLeft + downRight + 1 >= m_winAmount)
		return true;
	else if (upRight + downLeft + 1 >= m_winAmount)
		return true;

	return false;
}