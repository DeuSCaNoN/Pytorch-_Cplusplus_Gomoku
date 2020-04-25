#pragma once

#include <IPlayer.h>

#define ConvertToIndex(row, col, sideLength) (row * sideLength) + col

namespace GomokuUtils
{
	void ConvertToRowCol(int index, int sideLength, short& row, short& col);

	void TeachFromValueSet(bool bGenerate = false);

	void TrainSelfPlay();

	void TrainBluPig();

	short Evaluate();

	void DrawMatrix(char* matrix, int sideLength);

	struct PlayGeneratorCfg
	{
		unsigned short seed;
		unsigned int gameCount;
		bool bRandStart;
		bool bRandMoves;
		bool bSavePlayer1;
		bool bSavePlayer2;
		std::shared_ptr<Player::IPlayer> pPlayer1;
		std::shared_ptr<Player::IPlayer> pPlayer2;
		bool bPrint;
		int startMove;
	};
}
