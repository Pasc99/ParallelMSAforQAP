#pragma once
#include "SA.h"
#include "omp.h"
#include <thread>

class PMSA :
	public SA
{
private:
	const char* m_method = "PMSA";					// Method name
	// Hard to parallelize with members
	// So basically we use local variables
public:
	// Default constructor
	PMSA();
	// Copy constructor
	PMSA(const int& instance_num, const int& num_trial, const int& N_state);
	// Destructor
	~PMSA();

	int num_trial;									// Number of trials
	int N_state;									// Number of states
	std::vector<int> m_sol_gbest;					// Stores best solution each trial
	std::vector<int> f_best_all;					// Stores best cost from all trials
	double cost_mean;								// Mean of cost from all trials
	int cost_global;								// Best cost from all trials
	int cost_1trial;								// Best cost from one trial
	double cost_worst;								// Worst cost from all trials
	double R_error;									// Relative error

	// Reset after each trial
	void reset();
	// Initialize some variables
	void initialize();

	// Collect experiment results
	void result();

	// Thread-local random number generator
	double rnd(double min_arg, double max_arg);
	int rnd_int(int min_arg, int max_arg);
	
	// Output log file as results
	void output_log(const std::string& method);
	std::string file_dir;							// Directory of output file
	std::ofstream record_score;						// Output file

	// Initialize solution
	void permutation(std::vector<int>& sol, const int& dim);
	// Calculate cost (omp)
	int calculate_cost(std::vector<int>& X);
	// Calculate delta cost (omp)
	int calculate_delta_cost(std::vector<int>& sol, int& s1, int& s2);
	
	// 2-opt swap, return indexes of swapped elements
	std::tuple<int, int> swap_2opt();
	// Accept or reject new solution
	void SA_accept(std::vector<int>& sol, std::vector<int>& sol_best,
		int& cost, int& cost_best, int& s1, int& s2, double& T_curr);
	// Perform SA, return best solution and cost
	std::tuple<std::vector<int>, int> SA_2opt(std::vector<int>& sol);

	// Solve QAP with Parallel SA (main function)
	void SolveQAP();
	// Perform test trial before main trials
	void Test_trial();
};

