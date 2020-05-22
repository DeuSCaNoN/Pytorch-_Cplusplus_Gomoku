#include "pch.h"
#include "GomokuGame.h"
#include "GomokuUtils.h"

/*--------------------------------------------------------------*/

GomokuGame::GomokuGame(short sideLength, short winAmount)
	: m_sideLength(sideLength)
	, m_winAmount(winAmount)
	, m_boardLength(sideLength * sideLength)
	, m_movesPlayed(0)
	, m_lastMovePlayed(-1)
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
	m_lastMovePlayed = other.m_lastMovePlayed;

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
	m_lastMovePlayed = other.m_lastMovePlayed;

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
	m_lastMovePlayed = -1;
	m_playerTurn = true;
}

bool GomokuGame::IsBoardFull() const
{
	return m_movesPlayed == m_boardLength;
}

bool GomokuGame::IsMoveWinning(int index) const
{
	bool bBreakFirstPlayRule = false;
	bool bWinning = IsMoveWinning_(index, bBreakFirstPlayRule);
	if (bBreakFirstPlayRule && m_pGameBoard[index] == _P1SYMBOL_)
		return false;

	return bWinning;
}

bool GomokuGame::IsMoveWinning(short row, short col) const
{
	return IsMoveWinning(ConvertToIndex(row, col, m_sideLength));
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

int GomokuGame::GetLastMove() const
{
	return m_lastMovePlayed;
}

int GomokuGame::GetMovesPlayed() const
{
	return m_movesPlayed;
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

	bool bBreakFirstPlayRule = false;
	bool bWinning = IsMoveWinning_(index, bBreakFirstPlayRule);

	if (bBreakFirstPlayRule && m_playerTurn)
		m_winner = WinnerState_enum::P2;
	else if (bWinning)
		m_winner = m_playerTurn ? WinnerState_enum::P1 : WinnerState_enum::P2;
	else if (IsBoardFull())
		m_winner = WinnerState_enum::Draw;

	m_playerTurn = !m_playerTurn;
	m_lastMovePlayed = index;
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

std::pair<int, bool> GomokuGame::DirectionAmount_(
	int index,
	char symbol,
	std::function<int(int)> indexModifier,
	std::function<bool(int)> indexCheck) const
{
	int aboveAmount = 0;
	int testIndex = indexModifier(index);
	bool bBlocked = false;
	while (indexCheck(testIndex))
	{
		if (m_pGameBoard[testIndex] == symbol)
		{
			aboveAmount++;
			testIndex = indexModifier(testIndex);
		}
		else if (m_pGameBoard[testIndex] == _EMPTYSYMBOL_)
		{
			break;
		}
		else
		{
			bBlocked = true;
			break;
		}
	}

	return std::pair<int, bool>(aboveAmount, bBlocked);
}

bool GomokuGame::IsMoveWinning_(int index, bool& brokeFirstPlayRules) const
{
	char symbol = m_pGameBoard[index];
	if (symbol == _EMPTYSYMBOL_)
		return false;

	int sideLength = m_sideLength;
	int boardLength = m_boardLength;

	auto aboveModifier = [sideLength](int index) {return index - sideLength;};
	auto aboveCheck = [](int index) {return index >= 0; };
	std::future<std::pair<int, bool>> abovePromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, aboveModifier, aboveCheck);

	auto belowModifier = [sideLength](int index) {return index + sideLength;};
	auto belowCheck = [boardLength](int index) {return index < boardLength;};
	std::future<std::pair<int, bool>> belowPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, belowModifier, belowCheck);

	auto leftModifier = [](int index) {return index - 1; };
	auto leftCheck = [sideLength](int index) {return (index % sideLength) < (sideLength - 1);};
	std::future<std::pair<int, bool>> leftPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, leftModifier, leftCheck);

	auto rightModifier = [](int index) {return index + 1; };
	auto rightCheck = [sideLength](int index) {return (index % sideLength) != 0;};
	std::future<std::pair<int, bool>> rightPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, rightModifier, rightCheck);

	auto upLeftModifier = [sideLength](int index) {return index - sideLength - 1;};
	auto upLeftCheck = [sideLength](int index) {return index > 0 && (index % sideLength) < (sideLength - 1);};
	std::future<std::pair<int, bool>> upLeftPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, upLeftModifier, upLeftCheck);

	auto upRightModifier = [sideLength](int index) {return index - sideLength + 1; };
	auto upRightCheck = [sideLength](int index) {return index > 0 && (index % sideLength) != 0;};
	std::future<std::pair<int, bool>> upRightPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, upRightModifier, upRightCheck);

	auto downLeftModifier = [sideLength](int index) {return index + sideLength - 1; };
	auto downLeftCheck = [boardLength, sideLength](int index) {return index < boardLength && (index % sideLength) < (sideLength - 1);};
	std::future<std::pair<int, bool>> downLeftPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, downLeftModifier, downLeftCheck);

	auto downRightModifier = [sideLength](int index) {return index + sideLength + 1; };
	auto downRightCheck = [boardLength, sideLength](int index) {return index < boardLength && (index % sideLength) != 0;};
	std::future<std::pair<int, bool>> downRightPromise = std::async(&GomokuGame::DirectionAmount_, this, index, symbol, downRightModifier, downRightCheck);

	auto abovePair = abovePromise.get();
	auto belowPair = belowPromise.get();
	auto leftPair = leftPromise.get();
	auto rightPair = rightPromise.get();

	auto upLeftPair = upLeftPromise.get();
	auto upRightPair = upRightPromise.get();
	auto downLeftPair = downLeftPromise.get();
	auto downRightPair = downRightPromise.get();

	short freeThree = 0;
	short freeFour = 0;
	bool winner = false;

	int northSouth = abovePair.first + belowPair.first;
	int eastWest = leftPair.first + rightPair.first;
	int leftDiag = upLeftPair.first + downRightPair.first;
	int rightDiag = upRightPair.first + downLeftPair.first;

	if (!abovePair.second && !belowPair.second && northSouth + 1 == 3)
		freeThree++;

	if (!abovePair.second && !belowPair.second && northSouth + 1 == 4)
		freeFour++;

	if (!leftPair.second && !rightPair.second && eastWest + 1 == 3)
		freeThree++;

	if (!leftPair.second && !rightPair.second && eastWest + 1 == 4)
		freeFour++;

	if (!upLeftPair.second && !downRightPair.second && leftDiag + 1 == 3)
		freeThree++;

	if (!upLeftPair.second && !downRightPair.second && leftDiag + 1 == 4)
		freeFour++;

	if (!upRightPair.second && !downLeftPair.second && rightDiag + 1 == 3)
		freeThree++;

	if (!upRightPair.second && !downLeftPair.second && rightDiag + 1 == 4)
		freeFour++;

	if (northSouth + 1 >= m_winAmount)
	{
		if (northSouth >= m_winAmount)
			brokeFirstPlayRules = true;

		winner = true;
	}
	else if (eastWest + 1 >= m_winAmount)
	{
		if (eastWest >= m_winAmount)
			brokeFirstPlayRules = true;

		winner = true;
	}
	else if (leftDiag + 1 >= m_winAmount)
	{
		if (leftDiag >= m_winAmount)
			brokeFirstPlayRules = true;

		winner = true;
	}
	else if (rightDiag + 1 >= m_winAmount)
	{
		if (rightDiag >= m_winAmount)
			brokeFirstPlayRules = true;

		winner = true;
	}

	if (freeThree >= 2)
		brokeFirstPlayRules = true;
	else if (freeFour >= 2)
		brokeFirstPlayRules = true;

	return winner;
}