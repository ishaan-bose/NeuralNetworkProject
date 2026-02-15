#include "NNmath.hpp"
#include "Helper.hpp"

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
    /*
    ██████╗░███████╗░█████╗░██████╗░██╗███╗░░██╗░██████╗░
    ██╔══██╗██╔════╝██╔══██╗██╔══██╗██║████╗░██║██╔════╝░
    ██████╔╝█████╗░░███████║██║░░██║██║██╔██╗██║██║░░██╗░
    ██╔══██╗██╔══╝░░██╔══██║██║░░██║██║██║╚████║██║░░╚██╗
    ██║░░██║███████╗██║░░██║██████╔╝██║██║░╚███║╚██████╔╝
    ╚═╝░░╚═╝╚══════╝╚═╝░░╚═╝╚═════╝░╚═╝╚═╝░░╚══╝░╚═════╝░
    */
    double *W740 = loadCsvToMatrix("TrainingWeights/weights740.csv"); // 512x740
    double *W512 = loadCsvToMatrix("TrainingWeights/weights512.csv"); // 256x512
    double *W256 = loadCsvToMatrix("TrainingWeights/weights256.csv"); // 128x256
    double *W128 = loadCsvToMatrix("TrainingWeights/weights128.csv"); // 64x128
    double *W64 = loadCsvToMatrix("TrainingWeights/weights64.csv");   // 16x64
    double *W16 = loadCsvToMatrix("TrainingWeights/weights16.csv");   // 1x16 or 16x1 however you want to see it

    // NOTE: the names are slightly misleading, the bias arrays hold biases for the NEXT layer
    double *B740 = loadCsvToMatrix("TrainingWeights/biases740.csv");
    double *B512 = loadCsvToMatrix("TrainingWeights/biases512.csv");
    double *B256 = loadCsvToMatrix("TrainingWeights/biases256.csv");
    double *B128 = loadCsvToMatrix("TrainingWeights/biases128.csv");
    double *B64 = loadCsvToMatrix("TrainingWeights/biases64.csv");

    double B16;
    std::ifstream lastBiasFile("TrainingWeights/biases16.csv");
    lastBiasFile >> B16;
    lastBiasFile.close();

    rapidcsv::Document TrainingData("TrainingSet/test1.csv", rapidcsv::LabelParams(-1, -1));
    std::vector<std::string> positions = TrainingData.GetColumn<std::string>(0);
    std::vector<double> evals = TrainingData.GetColumn<double>(1);


    /*
██████╗░███████╗░█████╗░██╗░░░░░░█████╗░██████╗░░█████╗░████████╗██╗░█████╗░███╗░░██╗
██╔══██╗██╔════╝██╔══██╗██║░░░░░██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██╔══██╗████╗░██║
██║░░██║█████╗░░██║░░╚═╝██║░░░░░███████║██████╔╝███████║░░░██║░░░██║██║░░██║██╔██╗██║
██║░░██║██╔══╝░░██║░░██╗██║░░░░░██╔══██║██╔══██╗██╔══██║░░░██║░░░██║██║░░██║██║╚████║
██████╔╝███████╗╚█████╔╝███████╗██║░░██║██║░░██║██║░░██║░░░██║░░░██║╚█████╔╝██║░╚███║
╚═════╝░╚══════╝░╚════╝░╚══════╝╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚═╝░░░╚═╝░░░╚═╝░╚════╝░╚═╝░░╚══╝

░░░░██╗
░░░██╔╝
░░██╔╝░
░██╔╝░░
██╔╝░░░
╚═╝░░░░

██╗███╗░░██╗██╗████████╗██╗░█████╗░██╗░░░░░██╗███████╗░█████╗░████████╗██╗░█████╗░███╗░░██╗
██║████╗░██║██║╚══██╔══╝██║██╔══██╗██║░░░░░██║╚════██║██╔══██╗╚══██╔══╝██║██╔══██╗████╗░██║
██║██╔██╗██║██║░░░██║░░░██║███████║██║░░░░░██║░░███╔═╝███████║░░░██║░░░██║██║░░██║██╔██╗██║
██║██║╚████║██║░░░██║░░░██║██╔══██║██║░░░░░██║██╔══╝░░██╔══██║░░░██║░░░██║██║░░██║██║╚████║
██║██║░╚███║██║░░░██║░░░██║██║░░██║███████╗██║███████╗██║░░██║░░░██║░░░██║╚█████╔╝██║░╚███║
╚═╝╚═╝░░╚══╝╚═╝░░░╚═╝░░░╚═╝╚═╝░░╚═╝╚══════╝╚═╝╚══════╝╚═╝░░╚═╝░░░╚═╝░░░╚═╝░╚════╝░╚═╝░░╚══╝
    */

    // stands for gradient of a certain layer to update it in a mini batch or epoch or wtv
    double *grdntW740 = new double[512 * 740];
    double *grdntW512 = new double[256 * 512];
    double *grdntW256 = new double[128 * 256];
    double *grdntW128 = new double[64 * 128];
    double *grdntW64 = new double[16 * 64];
    double *grdntW16 = new double[16];

    double *grdntB740 = new double[512];
    double *grdntB512 = new double[256];
    double *grdntB256 = new double[128];
    double *grdntB128 = new double[64];
    double *grdntB64 = new double[16];
    double grdntB16 = 0.0;


    double *TempGrdntW740 = new double[512 * 740];
    double *TempGrdntW512 = new double[256 * 512];
    double *TempGrdntW256 = new double[128 * 256];
    double *TempGrdntW128 = new double[64 * 128];
    double *TempGrdntW64 = new double[16 * 64];
    double *TempGrdntW16 = new double[16];


    //NOTE: there is no TempGrdntB variables because the derivative vectors
    //calculated themselves ARE the temp gradient



    // stands for Activation for a certain layer

    // this is the input activation
    double *InputActivation = new double[740];

    // all the subsequent activations
    double *Actvtn512 = new double[512];
    double *Actvtn256 = new double[256];
    double *Actvtn128 = new double[128];
    double *Actvtn64 = new double[64];
    double *Actvtn16 = new double[16];

    //output activation
    double ActvtnOutput = 0.0;

    // stands for hyperbolic secant square of the neuron activation
    // which is the derivative of hyperbolic tangent
    double *Sech2_512 = new double[512];
    double *Sech2_256 = new double[256];
    double *Sech2_128 = new double[128];
    double *Sech2_64 = new double[64];
    double *Sech2_16 = new double[16];
    double Sech2_Output = 0.0;

    // stands for intermediate derivative
    double *IntermDer16 = new double[16];
    double *IntermDer64 = new double[64];
    double *IntermDer128 = new double[128];
    double *IntermDer256 = new double[256];
    double *IntermDer512 = new double[512];

    // stands for derivative of a certain layer, used to calculate gradient along with derivatives
    // of previous layer through backpropogation
    double derOutput = 0.0;
    double *der16 = new double[16];
    double *der64 = new double[64];
    double *der128 = new double[128];
    double *der256 = new double[256];
    double *der512 = new double[512];
    // no derivative vector for 740 cuz thats the input layer

    // miscellaneous variables

    //learning rate parameter, shouldnt be too small or big
    const double LearningRate = 0.005;
    const unsigned int MiniBatchSize = 1;
    double Cost = 0.0;
    double GroundTruth = 0.69'420'67;
    const double updateCoefficient = LearningRate/MiniBatchSize;







    /*
    std::fill(InputActivation, InputActivation + 740, 1.0);

    matrix_vector_multiply_blas(W740, InputActivation, Actvtn512, 512, 740);
    avx512_add(Actvtn512, B740, Actvtn512, 512);
    avx512_tanh_pade(Actvtn512, 512);
    avx512_sech2_from_tanh(Actvtn512, Sech2_512, 512);

    matrix_vector_multiply_blas(W512, Actvtn512, Actvtn256, 256, 512);
    avx512_add(Actvtn256, B512, Actvtn256, 256);
    avx512_tanh_pade(Actvtn256, 256);
    avx512_sech2_from_tanh(Actvtn256, Sech2_256, 256);

    matrix_vector_multiply_blas(W256, Actvtn256, Actvtn128, 128, 256);
    avx512_add(Actvtn128, B256, Actvtn128, 128);
    avx512_tanh_pade(Actvtn128, 128);
    avx512_sech2_from_tanh(Actvtn128, Sech2_128, 128);

    matrix_vector_multiply_blas(W128, Actvtn128, Actvtn64, 64, 128);
    avx512_add(Actvtn64, B128, Actvtn64, 64);
    avx512_tanh_pade(Actvtn64, 64);
    avx512_sech2_from_tanh(Actvtn64, Sech2_64, 64);

    matrix_vector_multiply_blas(W64, Actvtn64, Actvtn16, 16, 64);
    avx512_add(Actvtn16, B64, Actvtn16, 16);
    avx512_tanh_pade(Actvtn16, 16);
    avx512_sech2_from_tanh(Actvtn16, Sech2_16, 16);

    ActvtnOutput = std::tanh(B16 + avx512_dot_product(W16, Actvtn16, 16));
    Sech2_Output = 1.0 - ActvtnOutput*ActvtnOutput;

    derOutput = (ActvtnOutput - GroundTruth)*Sech2_Output;

    avx512_mul_scalar(W16, derOutput, IntermDer16, 16);
    avx512_hadamard_product(IntermDer16, Sech2_16, der16, 16);

    vector_matrix_multiply(der16, W64, IntermDer64, 16, 64);
    avx512_hadamard_product(IntermDer64, Sech2_64, der64, 64);

    vector_matrix_multiply(der64, W128, IntermDer128, 64, 128);
    avx512_hadamard_product(IntermDer128, Sech2_128, der128, 128);

    vector_matrix_multiply(der128, W256, IntermDer256, 128, 256);
    avx512_hadamard_product(IntermDer256, Sech2_256, der256, 256);

    vector_matrix_multiply(der256, W512, IntermDer512, 256, 512);
    avx512_hadamard_product(IntermDer512, Sech2_512, der512, 512);
    
    


    //██╗░░░██╗██████╗░██████╗░░█████╗░████████╗██╗███╗░░██╗░██████╗░
    //██║░░░██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║████╗░██║██╔════╝░
    //██║░░░██║██████╔╝██║░░██║███████║░░░██║░░░██║██╔██╗██║██║░░██╗░
    //██║░░░██║██╔═══╝░██║░░██║██╔══██║░░░██║░░░██║██║╚████║██║░░╚██╗
    //╚██████╔╝██║░░░░░██████╔╝██║░░██║░░░██║░░░██║██║░╚███║╚██████╔╝
    //░╚═════╝░╚═╝░░░░░╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░╚═╝╚═╝░░╚══╝░╚═════╝░

    //░██████╗░██████╗░░█████╗░██████╗░██╗███████╗███╗░░██╗████████╗
    //██╔════╝░██╔══██╗██╔══██╗██╔══██╗██║██╔════╝████╗░██║╚══██╔══╝
    //██║░░██╗░██████╔╝███████║██║░░██║██║█████╗░░██╔██╗██║░░░██║░░░
    //██║░░╚██╗██╔══██╗██╔══██║██║░░██║██║██╔══╝░░██║╚████║░░░██║░░░
    //╚██████╔╝██║░░██║██║░░██║██████╔╝██║███████╗██║░╚███║░░░██║░░░
    //░╚═════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═════╝░╚═╝╚══════╝╚═╝░░╚══╝░░░╚═╝░░░

    avx512_mul_scalar(Actvtn16, derOutput, TempGrdntW16, 16);
    avx512_outer_product(der16, 16, Actvtn64, 64, TempGrdntW64);
    avx512_outer_product(der64, 64, Actvtn128, 128, TempGrdntW128);
    avx512_outer_product(der128, 128, Actvtn256, 256, TempGrdntW256);
    avx512_outer_product(der256, 256, Actvtn512, 512, TempGrdntW512);
    avx512_outer_product(der512, 512, InputActivation, 740, TempGrdntW740);


    save_all_gradients_and_activations(
    TempGrdntW740, TempGrdntW512, TempGrdntW256, 
    TempGrdntW128, TempGrdntW64, TempGrdntW16,
    der512, der256, der128,
    der64, der16, derOutput,
    // Activation arrays
    InputActivation, Actvtn512, Actvtn256,
    Actvtn128, Actvtn64, Actvtn16,
    ActvtnOutput);


    */


    /*

    for (size_t batch_start = 0; batch_start < evals.size(); batch_start += MiniBatchSize)
    {
        size_t batch_end = std::min(batch_start + MiniBatchSize, evals.size());

        std::fill(grdntW740, grdntW740 + 512*740, 0.0);
        std::fill(grdntW512, grdntW512 + 256*512, 0.0);
        std::fill(grdntW256, grdntW256 + 128*256, 0.0);
        std::fill(grdntW128, grdntW128 + 64*128, 0.0);
        std::fill(grdntW64, grdntW64 + 16*64, 0.0);
        std::fill(grdntW16, grdntW16 + 1*16, 0.0);

        std::fill(grdntB740, grdntB740 + 512, 0.0);
        std::fill(grdntB512, grdntB512 + 256, 0.0);
        std::fill(grdntB256, grdntB256 + 128, 0.0);
        std::fill(grdntB128, grdntB128 + 64, 0.0);
        std::fill(grdntB64, grdntB64 + 16, 0.0);

        grdntB16 = 0.0;
        
        for (size_t i = batch_start; i < batch_end; ++i)
        {
            GroundTruth = evals[i];

            //██╗███╗░░██╗███████╗███████╗██████╗░███████╗███╗░░██╗░█████╗░██╗███╗░░██╗░██████╗░
            //██║████╗░██║██╔════╝██╔════╝██╔══██╗██╔════╝████╗░██║██╔══██╗██║████╗░██║██╔════╝░
            //██║██╔██╗██║█████╗░░█████╗░░██████╔╝█████╗░░██╔██╗██║██║░░╚═╝██║██╔██╗██║██║░░██╗░
            //██║██║╚████║██╔══╝░░██╔══╝░░██╔══██╗██╔══╝░░██║╚████║██║░░██╗██║██║╚████║██║░░╚██╗
            //██║██║░╚███║██║░░░░░███████╗██║░░██║███████╗██║░╚███║╚█████╔╝██║██║░╚███║╚██████╔╝
            //╚═╝╚═╝░░╚══╝╚═╝░░░░░╚══════╝╚═╝░░╚═╝╚══════╝╚═╝░░╚══╝░╚════╝░╚═╝╚═╝░░╚══╝░╚═════╝░

            FENtoVector(positions[i], InputActivation);

            matrix_vector_multiply_blas(W740, InputActivation, Actvtn512, 512, 740);
            avx512_add(Actvtn512, B740, Actvtn512, 512);
            avx512_tanh_pade(Actvtn512, 512);
            avx512_sech2_from_tanh(Actvtn512, Sech2_512, 512);

            matrix_vector_multiply_blas(W512, Actvtn512, Actvtn256, 256, 512);
            avx512_add(Actvtn256, B512, Actvtn256, 256);
            avx512_tanh_pade(Actvtn256, 256);
            avx512_sech2_from_tanh(Actvtn256, Sech2_256, 256);

            matrix_vector_multiply_blas(W256, Actvtn256, Actvtn128, 128, 256);
            avx512_add(Actvtn128, B256, Actvtn128, 128);
            avx512_tanh_pade(Actvtn128, 128);
            avx512_sech2_from_tanh(Actvtn128, Sech2_128, 128);

            matrix_vector_multiply_blas(W128, Actvtn128, Actvtn64, 64, 128);
            avx512_add(Actvtn64, B128, Actvtn64, 64);
            avx512_tanh_pade(Actvtn64, 64);
            avx512_sech2_from_tanh(Actvtn64, Sech2_64, 64);

            matrix_vector_multiply_blas(W64, Actvtn64, Actvtn16, 16, 64);
            avx512_add(Actvtn16, B64, Actvtn16, 16);
            avx512_tanh_pade(Actvtn16, 16);
            avx512_sech2_from_tanh(Actvtn16, Sech2_16, 16);

            ActvtnOutput = std::tanh(B16 + avx512_dot_product(W16, Actvtn16, 16));
            Sech2_Output = 1.0 - ActvtnOutput*ActvtnOutput;

            double curCost = 0.5*(GroundTruth - ActvtnOutput)*(GroundTruth - ActvtnOutput);
            Cost += curCost;

            //██████╗░░█████╗░░█████╗░██╗░░██╗██████╗░██████╗░░█████╗░██████╗░░█████╗░░██████╗░░█████╗░████████╗██╗░█████╗░███╗░░██╗
            //██╔══██╗██╔══██╗██╔══██╗██║░██╔╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝░██╔══██╗╚══██╔══╝██║██╔══██╗████╗░██║
            //██████╦╝███████║██║░░╚═╝█████═╝░██████╔╝██████╔╝██║░░██║██████╔╝██║░░██║██║░░██╗░███████║░░░██║░░░██║██║░░██║██╔██╗██║
            //██╔══██╗██╔══██║██║░░██╗██╔═██╗░██╔═══╝░██╔══██╗██║░░██║██╔═══╝░██║░░██║██║░░╚██╗██╔══██║░░░██║░░░██║██║░░██║██║╚████║
            //██████╦╝██║░░██║╚█████╔╝██║░╚██╗██║░░░░░██║░░██║╚█████╔╝██║░░░░░╚█████╔╝╚██████╔╝██║░░██║░░░██║░░░██║╚█████╔╝██║░╚███║
            //╚═════╝░╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝╚═╝░░░░░╚═╝░░╚═╝░╚════╝░╚═╝░░░░░░╚════╝░░╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░╚═╝░╚════╝░╚═╝░░╚══╝


            derOutput = (GroundTruth - ActvtnOutput)*(-1.0)*Sech2_Output;

            avx512_mul_scalar(W16, derOutput, IntermDer16, 16);
            avx512_hadamard_product(IntermDer16, Sech2_16, der16, 16);

            vector_matrix_multiply(der16, W64, IntermDer64, 16, 64);
            avx512_hadamard_product(IntermDer64, Sech2_64, der64, 64);

            vector_matrix_multiply(der64, W128, IntermDer128, 64, 128);
            avx512_hadamard_product(IntermDer128, Sech2_128, der128, 128);

            vector_matrix_multiply(der128, W256, IntermDer256, 128, 256);
            avx512_hadamard_product(IntermDer256, Sech2_256, der256, 256);

            vector_matrix_multiply(der256, W512, IntermDer512, 256, 512);
            avx512_hadamard_product(IntermDer512, Sech2_512, der512, 512);


            //██╗░░░██╗██████╗░██████╗░░█████╗░████████╗██╗███╗░░██╗░██████╗░
            //██║░░░██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║████╗░██║██╔════╝░
            //██║░░░██║██████╔╝██║░░██║███████║░░░██║░░░██║██╔██╗██║██║░░██╗░
            //██║░░░██║██╔═══╝░██║░░██║██╔══██║░░░██║░░░██║██║╚████║██║░░╚██╗
            //╚██████╔╝██║░░░░░██████╔╝██║░░██║░░░██║░░░██║██║░╚███║╚██████╔╝
            //░╚═════╝░╚═╝░░░░░╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░╚═╝╚═╝░░╚══╝░╚═════╝░

            //░██████╗░██████╗░░█████╗░██████╗░██╗███████╗███╗░░██╗████████╗
            //██╔════╝░██╔══██╗██╔══██╗██╔══██╗██║██╔════╝████╗░██║╚══██╔══╝
            //██║░░██╗░██████╔╝███████║██║░░██║██║█████╗░░██╔██╗██║░░░██║░░░
            //██║░░╚██╗██╔══██╗██╔══██║██║░░██║██║██╔══╝░░██║╚████║░░░██║░░░
            //╚██████╔╝██║░░██║██║░░██║██████╔╝██║███████╗██║░╚███║░░░██║░░░
            //░╚═════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═════╝░╚═╝╚══════╝╚═╝░░╚══╝░░░╚═╝░░░

            avx512_mul_scalar(Actvtn16, derOutput, TempGrdntW16, 16);
            avx512_outer_product(der16, 16, Actvtn64, 64, TempGrdntW64);
            avx512_outer_product(der64, 64, Actvtn128, 128, TempGrdntW128);
            avx512_outer_product(der128, 128, Actvtn256, 256, TempGrdntW256);
            avx512_outer_product(der256, 256, Actvtn512, 512, TempGrdntW512);
            avx512_outer_product(der512, 512, InputActivation, 740, TempGrdntW740);

            avx512_fma(TempGrdntW16, grdntW16, updateCoefficient, grdntW16, 1*16);
            avx512_fma(TempGrdntW64, grdntW64, updateCoefficient, grdntW64, 16*64);
            avx512_fma(TempGrdntW128, grdntW128, updateCoefficient, grdntW128, 64*128);
            avx512_fma(TempGrdntW256, grdntW256, updateCoefficient, grdntW256, 128*256);
            avx512_fma(TempGrdntW512, grdntW512, updateCoefficient, grdntW512, 256*512);
            avx512_fma(TempGrdntW740, grdntW740, updateCoefficient, grdntW740, 512*740);

            avx512_fma(der512, grdntB740, updateCoefficient, grdntB740, 512);
            avx512_fma(der256, grdntB512, updateCoefficient, grdntB512, 256);
            avx512_fma(der128, grdntB256, updateCoefficient, grdntB256, 128);
            avx512_fma(der64, grdntB128, updateCoefficient, grdntB128, 64);
            avx512_fma(der16, grdntB64, updateCoefficient, grdntB64, 16);
            grdntB16 += updateCoefficient*derOutput;

            std::cout << "value of position " << i << " is " << ActvtnOutput << "\n";
            
        }

        //███╗░░░███╗██╗███╗░░██╗██╗  ██████╗░░█████╗░████████╗░█████╗░██╗░░██╗
        //████╗░████║██║████╗░██║██║  ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██║░░██║
        //██╔████╔██║██║██╔██╗██║██║  ██████╦╝███████║░░░██║░░░██║░░╚═╝███████║
        //██║╚██╔╝██║██║██║╚████║██║  ██╔══██╗██╔══██║░░░██║░░░██║░░██╗██╔══██║
        //██║░╚═╝░██║██║██║░╚███║██║  ██████╦╝██║░░██║░░░██║░░░╚█████╔╝██║░░██║
        //╚═╝░░░░░╚═╝╚═╝╚═╝░░╚══╝╚═╝  ╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░░╚════╝░╚═╝░░╚═╝

        //██╗░░░██╗██████╗░██████╗░░█████╗░████████╗███████╗
        //██║░░░██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
        //██║░░░██║██████╔╝██║░░██║███████║░░░██║░░░█████╗░░
        //██║░░░██║██╔═══╝░██║░░██║██╔══██║░░░██║░░░██╔══╝░░
        //╚██████╔╝██║░░░░░██████╔╝██║░░██║░░░██║░░░███████╗
        //░╚═════╝░╚═╝░░░░░╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░╚══════╝

        avx512_sub(W740, grdntW740, W740, 512*740);
        avx512_sub(W512, grdntW512, W512, 256*512);
        avx512_sub(W256, grdntW256, W256, 128*256);
        avx512_sub(W128, grdntW128, W128, 64*128);
        avx512_sub(W64, grdntW64, W64, 16*64);
        avx512_sub(W16, grdntW16, W16, 1*16);

        avx512_sub(B740, grdntB740, B740, 512);
        avx512_sub(B512, grdntB512, B512, 256);
        avx512_sub(B256, grdntB256, B256, 128);
        avx512_sub(B128, grdntB128, B128, 64);
        avx512_sub(B64, grdntB64, B64, 16);
        B16 -= grdntB16;
    }

    Cost /= evals.size();

    std::cout << "Average Cost is: " << Cost << "\n";

    




    //░██████╗░█████╗░██╗░░░██╗██╗███╗░░██╗░██████╗░  ████████╗░█████╗░  ███████╗██╗██╗░░░░░███████╗
    //██╔════╝██╔══██╗██║░░░██║██║████╗░██║██╔════╝░  ╚══██╔══╝██╔══██╗  ██╔════╝██║██║░░░░░██╔════╝
    //╚█████╗░███████║╚██╗░██╔╝██║██╔██╗██║██║░░██╗░  ░░░██║░░░██║░░██║  █████╗░░██║██║░░░░░█████╗░░
    //░╚═══██╗██╔══██║░╚████╔╝░██║██║╚████║██║░░╚██╗  ░░░██║░░░██║░░██║  ██╔══╝░░██║██║░░░░░██╔══╝░░
    //██████╔╝██║░░██║░░╚██╔╝░░██║██║░╚███║╚██████╔╝  ░░░██║░░░╚█████╔╝  ██║░░░░░██║███████╗███████╗
    //╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░╚═╝╚═╝░░╚══╝░╚═════╝░  ░░░╚═╝░░░░╚════╝░  ╚═╝░░░░░╚═╝╚══════╝╚══════╝


    WriteMatrixToCsv("TrainingWeights/weights740.csv", W740, 512, 740);
    WriteMatrixToCsv("TrainingWeights/weights512.csv", W512, 256, 512);
    WriteMatrixToCsv("TrainingWeights/weights256.csv", W256, 128, 256);
    WriteMatrixToCsv("TrainingWeights/weights128.csv", W128, 64, 128);
    WriteMatrixToCsv("TrainingWeights/weights64.csv", W64, 16, 64);
    WriteMatrixToCsv("TrainingWeights/weights16.csv", W16, 1, 16);

    // refer to the NOTE that is there when loading these arrays
    // for understanding why the next layers size is used for writing current layer
    // or else youll be confused as to why biases740 has 512 elements
    WriteMatrixToCsv("TrainingWeights/biases740.csv", B740, 1, 512);
    WriteMatrixToCsv("TrainingWeights/biases512.csv", B512, 1, 256);
    WriteMatrixToCsv("TrainingWeights/biases256.csv", B256, 1, 128);
    WriteMatrixToCsv("TrainingWeights/biases128.csv", B128, 1, 64);
    WriteMatrixToCsv("TrainingWeights/biases64.csv", B64, 1, 16);

    std::ofstream lastBiasFileWrite("TrainingWeights/biases16.csv");
    lastBiasFileWrite << (B16); // Corrected to use lastBiasFileWrite
    lastBiasFileWrite.close();

    */

    /*
    ░█████╗░██╗░░░░░███████╗░█████╗░███╗░░██╗██╗░░░██╗██████╗░
    ██╔══██╗██║░░░░░██╔════╝██╔══██╗████╗░██║██║░░░██║██╔══██╗
    ██║░░╚═╝██║░░░░░█████╗░░███████║██╔██╗██║██║░░░██║██████╔╝
    ██║░░██╗██║░░░░░██╔══╝░░██╔══██║██║╚████║██║░░░██║██╔═══╝░
    ╚█████╔╝███████╗███████╗██║░░██║██║░╚███║╚██████╔╝██║░░░░░
    ░╚════╝░╚══════╝╚══════╝╚═╝░░╚═╝╚═╝░░╚══╝░╚═════╝░╚═╝░░░░░
    */

    delete[] W740;
    delete[] W512;
    delete[] W256;
    delete[] W128;
    delete[] W64;
    delete[] W16;

    delete[] B740;
    delete[] B512;
    delete[] B256;
    delete[] B128;
    delete[] B64;

    delete[] grdntW740;
    delete[] grdntW512;
    delete[] grdntW256;
    delete[] grdntW128;
    delete[] grdntW64;
    delete[] grdntW16;

    delete[] grdntB740;
    delete[] grdntB512;
    delete[] grdntB256;
    delete[] grdntB128;
    delete[] grdntB64;

    delete[] TempGrdntW740;
    delete[] TempGrdntW512;
    delete[] TempGrdntW256;
    delete[] TempGrdntW128;
    delete[] TempGrdntW64;
    delete[] TempGrdntW16;

    delete[] InputActivation;
    delete[] Actvtn512;
    delete[] Actvtn256;
    delete[] Actvtn128;
    delete[] Actvtn64;
    delete[] Actvtn16;

    delete[] Sech2_512;
    delete[] Sech2_256;
    delete[] Sech2_128;
    delete[] Sech2_64;
    delete[] Sech2_16;

    delete[] IntermDer16;
    delete[] IntermDer64;
    delete[] IntermDer128;
    delete[] IntermDer256;
    delete[] IntermDer512;

    delete[] der16;
    delete[] der64;
    delete[] der128;
    delete[] der256;
    delete[] der512;

    return 0;
}