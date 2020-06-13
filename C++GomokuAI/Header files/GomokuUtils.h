#pragma once

#include <IPlayer.h>

#define ConvertToIndex(row, col, sideLength) (row * sideLength) + col

namespace GomokuUtils
{
	void ConvertToRowCol(int index, int sideLength, short& row, short& col);

	void HumanPlay(bool bHumanSide);

	void TeachFromValueSet(bool bGenerate = false);

	void TrainSelfPlay(bool bLoop = true, short loopCount = 30);

	void TrainBluPig(bool bLoop = true, short loopCount = 30);

	void MixedTraining();

	void Evaluate(std::shared_ptr<Player::IPlayer> pAgent1, std::shared_ptr<Player::IPlayer> pAgent2);

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
