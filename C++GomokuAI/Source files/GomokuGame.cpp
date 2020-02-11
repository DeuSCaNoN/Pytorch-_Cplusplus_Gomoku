#include "pch.h"
#include "GomokuGame.h"
#include "GomokuUtils.h"

#include <string.h>
#include <thread>
#include <future>
#include <iostream>

/*--------------------------------------------------------------*/

GomokuGame::GomokuGame(short sideLength, short winAmount)
	: m_sideLength(sideLength)
	, m_winAmount(winAmount)
	, m_boardLength(sideLength * sideLength)
	, m_movesPlayed(0)
	, m_winner(WinnerState_enum::None)
	, m_playerTurn(true)
{
	m_pGameBoard = new char[m_boardLength];
	m_pLegalMoves = new int[m_boardLength];

	for (int i = 0; i < m_boardLength; i++)
	{
		m_pGameBoard[i] = _EMPTYSYMBOL_;
		m_pLegalMoves[i] = i;
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

	m_pGameBoard = new char[m_boardLength];
	m_pLegalMoves = new int[m_boardLength - m_movesPlayed];
	memcpy(m_pGameBoard, other.m_pGameBoard, m_boardLength * sizeof(char));
	memcpy(m_pLegalMoves, other.m_pLegalMoves, (m_boardLength - m_movesPlayed) * sizeof(int));
}

GomokuGame::~GomokuGame()
{
	delete m_pGameBoard;
	delete m_pLegalMoves;
}

void GomokuGame::operator=(GomokuGame const& other)
{
	m_sideLength = other.m_sideLength;
	m_winAmount = other.m_winAmount;
	m_boardLength = m_sideLength * m_sideLength;
	m_movesPlayed = other.m_movesPlayed;
	m_winner = other.m_winner;
	m_playerTurn = other.m_playerTurn;

	delete m_pGameBoard;
	delete m_pLegalMoves;
	m_pGameBoard = new char[m_boardLength];
	m_pLegalMoves = new int[m_boardLength - m_movesPlayed];
	memcpy(m_pGameBoard, other.m_pGameBoard, m_boardLength * sizeof(char));
	memcpy(m_pLegalMoves, other.m_pLegalMoves, (m_boardLength - m_movesPlayed) * sizeof(int));
}

/*--------------------------------------------------------------*/

void GomokuGame::ResetBoard()
{
	for (int i = 0; i < m_boardLength; i++)
	{
		m_pGameBoard[i] = _EMPTYSYMBOL_;
		m_pLegalMoves[i] = i;
	}

	m_winner = WinnerState_enum::None;
	m_movesPlayed = 0;
	m_playerTurn = true;
}

bool GomokuGame::IsBoardFull() const
{
	return m_movesPlayed == m_boardLength;
}

bool GomokuGame::IsMoveWinning(int index) const
{
	return IsMoveWinning_(index);
}

bool GomokuGame::IsMoveWinning(short row, short col) const
{
	return IsMoveWinning_(ConvertToIndex(row, col, m_sideLength));
}

char* GomokuGame::GetBoard() const
{
	return m_pGameBoard;
}

char** GomokuGame::GetMatrix() const
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

		matrix[row][col] = m_pGameBoard[i];
	}

	return matrix;
}

bool GomokuGame::GetPlayerTurn() const
{
	return m_playerTurn;
}

short GomokuGame::GetSideLength() const
{
	return m_sideLength;
}

int* GomokuGame::GetLegalMoves(int& size) const
{
	size = m_boardLength - m_movesPlayed;
	return m_pLegalMoves;
}

bool GomokuGame::PlayMove(int index)
{
	if (m_pGameBoard[index] != _EMPTYSYMBOL_)
		return false;

	m_pGameBoard[index] = m_playerTurn ? _P1SYMBOL_ : _P2SYMBOL_;
	int legalMovesSize = m_boardLength - m_movesPlayed;
	bool bTakenOut = false;
	for (int i = 0; i < legalMovesSize; i++)
	{
		if (m_pLegalMoves[i] != index && bTakenOut)
			m_pLegalMoves[i - 1] = m_pLegalMoves[i];
		else if (m_pLegalMoves[i] == index)
			bTakenOut = true;
	}

	m_movesPlayed++;

	if (IsMoveWinning_(index))
		m_winner = m_playerTurn ? WinnerState_enum::P1 : WinnerState_enum::P2;
	else if (IsBoardFull())
		m_winner = WinnerState_enum::Draw;

	m_playerTurn = !m_playerTurn;
	return true;
}

bool GomokuGame::PlayMove(short row, short col)
{
	return PlayMove(ConvertToIndex(row, col, m_sideLength));
}

WinnerState_enum GomokuGame::GetGameWinState() const
{
	return m_winner;
}

/*--------------------------------------------------------------*/

void GomokuGame::FreeCurrentBoard_()
{
	delete m_pGameBoard;
}

int GomokuGame::DirectionAmount_(
	int index,
	char symbol,
	std::function<int(int)> indexModifier,
	std::function<bool(int)> indexCheck) const
{
	int aboveAmount = 0;
	int testIndex = indexModifier(index);
	while (indexCheck(testIndex))
	{
		if (m_pGameBoard[testIndex] == symbol)
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
	char symbol = m_pGameBoard[index];
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