#pragma once
#include "CudaPMSA.h"

int main(int argc, char* argv[])
{
	// Defalut argument:
	int instance_num = 0;
	int num_trial = 30;
	int N_state = 16;

	// Parsing argument:
	ParsingArg(argc, argv, instance_num, num_trial, N_state);
	//SA sa;
	//sa.generate_initX_auto(i);
	CudaPMSA pmsa(instance_num, num_trial, N_state);
	//pmsa.SolveQAP_record();
	//pmsa.Test_trial();
	pmsa.SolveQAP();
	return 0;
}

// __global__	Called by the CPU, and runs on the GPU
// __device__	Called by the GPU, and runs on the GPU
// __host__		Normal CPU function