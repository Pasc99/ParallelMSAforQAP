#pragma once

#define _USE_MATH_DEFINES

#include <iostream>
#include <ios>
#include <random>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <limits>
#include <numeric>
#include <filesystem>
#include <tuple>
#include <regex>
#include <omp.h>

class SA
{
private:
	std::random_device rng;
	std::mt19937_64 mt;
public:
	SA();
	~SA();

	// random number generator
	double rnd(double min_arg, double max_arg);
	int rnd_int(int min_arg, int max_arg);

	// QAP matrix
	std::vector<std::vector<int>> Distance;
	std::vector<std::vector<int>> Flow;
	
	// SA param
	std::string instance;
	int num_trial;				// Number of trials
	int N_state;				// Number of initial states
	double T_curr;				// Current Temp
	double T_0;					// Initial Temp
	double alpha;				// Cooling coefficent
	double T_end;				// Final Temp
	int num_re;					// Number of repetation
	int dim;					// Dimension of the instance 
	int answer;					// Known optimum (from QAPLIB)
	double success;				// Optimum reached rate

	// Generate initial state as txt for recording annealing curve
	// Not used in the final paper
	void generate_initX();
	void generate_initX_auto(const int& instance_num);

	// Read initial state from txt
	void read_trialset();

	// Buffer for initial states
	std::vector<std::vector<int>> buffer_trialset;

	// Read Distance and Flow matrix from QAPLIB dat file
	// Notice: some original QAPLIB dat files might contain known optimum in the first line
	// In this case, the optimum should be deleted (For example: tai25a.dat)
	void read_DF();

	// Print Distance and Flow matrix (for testing)
	void print_DF(const std::vector<std::vector<int>>& Distance, const std::vector<std::vector<int>>& Flow);

	// Calculate cost of the solution for QAP
	int calculate_cost(std::vector<int>& X);
	// Calculate delta cost of the solution for QAP
	int calculate_delta_cost(std::vector<int>& sol, int& s1, int& s2);
	// Counting success rate
	void success_counter(int&);

	// Set parameters of SA
	// (problem_name, T_0, T_end, K, known_optimum)
	void set_param(const char* name, double t0_arg, double T_end_arg, int K_arg, uint32_t answer_arg);

	// Select which instance to solve manually
	void choose_instance();
	// Select which instance to solve via instance index, which can be changed in the function
	void choose_instance_auto(const int& instance_num);
	// Select which instance to generate trialset, not used in final
	void choose_instance_lite();

	// Displaying current progress on the console
	void display_progress(const int& trial, const int& num_trial);
	// Get current time for logging
	std::string getTimeStr();
	// Get current date for logging
	std::string getDateStr();
};

// Error message for command line argument
void PrintErrorMsg();
// Parsing command line argument
int ParsingArg(int argc, char* argv[], int& instance_num, int& num_trial, int& N_state);
