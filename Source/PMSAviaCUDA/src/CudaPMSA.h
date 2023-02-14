#pragma once
#include "SA.h"
#include "CudaSA.h"

class CudaPMSA :
	public SA
{
private:
	std::vector<int> m_sol_gbest;					// Best solution for each trial
	std::string m_method = "CudaPMSA";				// Method name
public:
	// Default constructor
	CudaPMSA();
	// Copy constructor
	CudaPMSA(const int& instance_num, const int& num_trial, const int& N_state);
	// Destructor
	~CudaPMSA();

	int num_trial;									// Number of trials
	int N_state;									// Number of initial states

	size_t block_size;								// number of threads per block

	// Flatten D, F matrix:
	int* FDistance;									// Flattened Distance matrix
	int* FFlow;										// Flattened Flow matrix

	std::vector<int> f_best_all;					// Stores best cost for each trial
	double cost_mean;								// Mean of cost for all trials
	int cost_global;								// Best cost for all trials
	int cost_1trial;								// Best cost for one trial
	double cost_worst;								// Worst cost for all trials
	double R_error;									// Relative error

	// Resets nothing
	void reset();
	// Initialize some variables
	void initialize();

	// Collect experiment results
	void result();

	// Output log file as results
	void output_log(const std::string& method);
	std::string file_dir;							// Directory of output file
	std::ofstream record_score;						// Output file

	// Flatten 2D vector to 1D array
	void FlattenArray(int* dst, const std::vector<std::vector<int>>& src, int row, int col);
	// Restore 1D array to 2D vector
	void RestoreVector(std::vector<std::vector<int>>& dst, int* src, int row, int col);
	// Solve QAP with Parallel SA (main function)
	void SolveQAP();
	// Perform test trial before main trials
	void Test_trial();
};

