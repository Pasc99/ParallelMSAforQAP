#include "SA.h"
#include "SMSA.h"
#include "PMSA.h"

int main(int argc, char* argv[])
{
	// Defalut argument:
	int instance_num = 0;
	int num_trial = 30;
	int N_state = 12;
	
	// Parsing argument:
	ParsingArg(argc, argv, instance_num, num_trial, N_state);
	
	// Comment out the method you don't want to use before compiling.
	//SMSA smsa(instance_num, num_trial, N_state);
	//smsa.SolveQAP();
	PMSA pmsa(instance_num, num_trial, N_state);
	pmsa.SolveQAP();

	//std::cout << "Press any key to exit.";
	//std::cin.get();
	return 0;
}