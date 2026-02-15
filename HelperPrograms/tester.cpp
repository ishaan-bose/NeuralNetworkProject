#include <iostream>
#include "rapidcsv.h"


void FENtoVector(const std::string &fen, double *res)
{
    std::istringstream iss(fen);
    std::string first, second, third;
    iss >> first >> second >> third;
    // second is kinda useless here

    short position = 0;

    std::fill(res, res + 740, 0.0);

    // Iterate through the first substring
    for (size_t i = 0; i < first.size(); ++i)
    {
        char piece = first[i];

        switch (piece)
        {
        case 'r':
            res[0 * 64 + position] = 1.0;
            position += 1;
            break;
        case 'n':
            res[1 * 64 + position] = 1.0;
            position += 1;
            break;
        case 'b':
            res[2 * 64 + position] = 1.0;
            position += 1;
            break;
        case 'q':
            res[3 * 64 + position] = 1.0;
            position += 1;
            break;
        case 'k':
            res[4 * 64 + position] = 1.0;
            position += 1;
            break;
        case 'R':
            res[5 * 64 + position] = 1.0;
            position += 1;
            break;
        case 'N':
            res[6 * 64 + position] = 1.0;
            position += 1;
            break;
        case 'B':
            res[7 * 64 + position] = 1.0;
            position += 1;
            break;
        case 'Q':
            res[8 * 64 + position] = 1.0;
            position += 1;
            break;
        case 'K':
            res[9 * 64 + position] = 1.0;
            position += 1;
            break;
        case 'p':
            res[640 + position - 8] = 1.0;
            position += 1;
            break;
        case 'P':
            res[688 + position - 8] = 1.0;
            position += 1;
            break;
        case '1':
            position += 1;
            break;
        case '2':
            position += 2;
            break;
        case '3':
            position += 3;
            break;
        case '4':
            position += 4;
            break;
        case '5':
            position += 5;
            break;
        case '6':
            position += 6;
            break;
        case '7':
            position += 7;
            break;
        case '8':
            position += 8;
            break;
        case '/':
            // do nothing
            break;
        default:
            break;
        }
    }

    for (size_t i = 0; i < third.size(); ++i)
    {
        char piece = third[i];
        switch (piece)
        {
        case 'K':
            res[736] = 1.0;
            break;
        case 'Q':
            res[737] = 1.0;
            break;
        case 'k':
            res[738] = 1.0;
            break;
        case 'q':
            res[739] = 1.0;
            break;
        default:
            break;
        }
    }
}




int main()
{

    rapidcsv::Document TrainingData("../TrainingSet/test1.csv", rapidcsv::LabelParams(-1, -1));
    std::vector<std::string> positions = TrainingData.GetColumn<std::string>(0);
    std::vector<double> evals = TrainingData.GetColumn<double>(1);

    double *InputActivation = new double[740];

    for (size_t i = 0; i < 10; i++)
    {
        FENtoVector(positions[i], InputActivation);
        for (size_t j = 0; j < 739; j++)
        {
            std::cout << InputActivation[j] << ",";
        }

        std::cout << InputActivation[739] << "\n" << evals[i] << "\n\n";
    }

    return 0;
}