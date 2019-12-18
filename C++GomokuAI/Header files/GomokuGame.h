#pragma once
#include <functional>

enum WinnerState_enum
{
	None = 0,
	P1 = 1,
	P2 = 2,
	Draw = 3
};

class GomokuGame
{
public:
	GomokuGame(short sideLength, short winAmount);
	~GomokuGame();

	GomokuGame(GomokuGame const& other); // Copy constructor
	void operator=(GomokuGame const& other); // Copy operator

	/*--------------------------------------------------------------*/

	void ResetBoard();
	bool IsBoardFull() const;
	bool IsMoveWinning(int index);
	bool IsMoveWinning(short row, short col);

	bool PlayMove(short row, short col);

	char** GetMatrix();

	WinnerState_enum GetGameWinState() const;
private:

	void FreeCurrentBoard_();
	void UpdateWinBoard_(int index);

	bool IsMoveWinning_(int index) const;
	int DirectionAmount_(
		int index,
		char symbol,
		std::function<int(int)> indexModifier,
		std::function<bool(int)> indexCheck) const;

	short m_sideLength;
	short m_winAmount;
	int m_boardLength;
	char* m_gameBoard;
	int m_movesPlayed;

	WinnerState_enum m_winner;
	bool m_playerTurn;
};

