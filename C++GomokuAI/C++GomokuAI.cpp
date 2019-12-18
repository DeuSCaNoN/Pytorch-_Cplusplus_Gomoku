// C++GomokuAI.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>

#include <GomokuGame.h>

void DrawMatrix(char** matrix, int sideLength)
{
	std::cout << "\n";
	for (int i = 0; i < sideLength; i++)
	{
		for (int j = 0; j < sideLength; j++)
		{
			char output = 0;
			switch (matrix[i][j])
			{
			case 0:
				break;
			case 1:
				output = '1';
				break;
			case 2:
				output = '2';
			}
			std::cout << output << " | ";
		}
		std::cout << "\n";
	}
}

int main()
{
	GomokuGame game = GomokuGame(5, 3);
	game.PlayMove(0, 2);
	game.PlayMove(0, 0);
	game.PlayMove(0, 3);
	game.PlayMove(2, 2);
	game.PlayMove(1, 4);
	game.PlayMove(4, 0);
	game.PlayMove(1, 3);
	game.PlayMove(4, 3);
	
	char** matrix = game.GetMatrix();
	DrawMatrix(matrix, 5);
	return game.IsMoveWinning(0, 4);
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
