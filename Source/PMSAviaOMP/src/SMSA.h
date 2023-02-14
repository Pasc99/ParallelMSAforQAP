#pragma once
#include "SA.h"

class SMSA :
	public SA
{
private:
	std::vector<std::vector<int>> sol;				// Current Solution
	std::vector<std::vector<int>> sol_best;			// Best solution
	std::vector<int> cost;							// Current cost
	std::vector<int> cost_previous;					// Previous cost
	std::vector<double> cost_delta_rec;				// Records of Cost difference
	std::vector<double> prob_accept;				// probablity to accept
	std::vector<int> cost_best;						// Current best cost
	std::vector<std::vector<int>> sol_next;			// Next Solution
	std::vector<int> delta_cost;					// Difference of cost
	// 2-opt method:
	std::vector<int> s1;							// First element to be swapped
	std::vector<int> s2;							// Second element to be swapped
	const char* m_method = "SMSA";					// Method name
public:
	// Default constructor
	SMSA();
	// Copy constructor
	SMSA(const int& instance_num, const int& num_trial, const int& N_state);
	// Destructor
	~SMSA();
	// Initialize some variables
	void initialize();

	int num_trial;									// Number of trials
	int N_state;									// Number of states
	std::vector<int> f_best_all;					// Stores best cost from all trials
	double cost_mean;								// Mean of cost from all trials
	int cost_global;								// Best cost from all trials
	int cost_1trial;								// Best cost from one trial
	double cost_worst;								// Worst cost from all trials
	double R_error;									// Relative error
	

	// Reset after each trial
	void reset();						
	// Collect experiment results
	void result();

	// Output log file as results
	void output_log(const std::string& method);
	std::string file_dir;							// Directory of output file
	std::ofstream record_score;						// Output file

	//////////Functions for SA:////////

	// Initialize solution
	void permutation(std::vector<int>& sol, const int& dim);
	// 2-opt method to generate new solution
	void swap_2opt();
	// Accept or reject new solution
	void SA_accept();
	
	//////////Functions for experiments:////////

	// Solve QAP (main function)
	void SolveQAP();
	// Perform test trial before main trials
	void Test_trial();
};
